# Isaac Sim 상에서 걷기·일어서기 정책 학습 — 설계

작성 2026-08-25. 상태: **섹션 1~4 확정, 섹션 5~8 미작성.**

## 목표

**1차 과업: 시뮬레이터 안에서 걷기와 일어서기를 각각 성공하는 정책을 학습으로 얻는다.**
논문(arXiv 2404.05932)의 RL 결과를 Isaac Sim / Isaac Lab에서 재현하는 것이 기준선이다.

최종 목표는 실기 이식이다. 이 전제가 관측 설계를 강하게 구속한다(섹션 3).

### 정해둔 방향

| 항목 | 결정 | 근거 |
|---|---|---|
| 방법론 | 온폴리시 RL (PPO) | 논문과 동일. 별도 데이터 수집 단계가 없다 — 병렬 env 롤아웃이 곧 데이터다 |
| 종착점 | 실기 이식이 최종 목표 | 관측을 논문의 40차원으로 제한하고 DR·액추에이터 지연을 처음부터 넣는다 |
| 정책 구조 | walk / stand 2개 분리 + FSM 전환 | 논문 방식. 실기 배선(`src/rl_walk_and_stand.py`)이 이미 존재한다 |
| Isaac Lab 워크플로 | Direct (`DirectRLEnv`) | 논문 원본이 IsaacGymEnvs Ant 변형으로 보이고, 40차원 이력 관측이 비표준이라 직접 다루는 편이 명확하다 |

### Direct(A) → Manager-based(B) 이주 가능성

나중에 B로 갈아탈 수 있게 설계한다. 조건은 하나다:

**보상·종료·관측 계산을 `self`를 건드리지 않는 순수 함수로 `chair_rl/mdp.py`에 둔다.**
Direct env는 그 함수들을 부르는 얇은 껍데기가 된다. B로 갈 때 manager term은
`(env, ...)` 래퍼 한 줄로 같은 함수를 부르면 된다.

그대로 옮겨가는 것: 로봇 에셋 정의, 도메인 랜덤화(`DirectRLEnvCfg.events`가
manager-based와 **같은** `EventManager`/`EventTermCfg` API를 쓴다 —
`IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:151`), 보상·종료 수식,
ONNX 계약, 평가 하네스, FSM 임계값.

다시 쓰는 것: env 클래스 1개 → cfg 데이터클래스 묶음. 파일 하나 분량.

**주의 지점:** B의 `ObservationTermCfg.history_length`
(`manager_term_cfg.py:184`)는 자체 규약으로 flatten한다. 어긋나면 ONNX 입력
순서가 바뀌어 실기 코드와 조용히 틀어진다. 섹션 3의 레이아웃 테스트가 안전망이다.

## 현재 상태

### 레포에 있는 것

- `src/` — ROS 실기 코드. 학습 코드 아님, 배포 코드다.
- `models/walk.onnx`, `models/stand.onnx` — 논문의 학습 결과물. **비교 기준선.**
- `mjcf/chair.xml` — MuJoCo 모델. Isaac에서 USD로 변환해 쓴다.
- `isaac/chair_sim.py` — Isaac Lab 재생기. ONNX 정책 또는 고정 키프레임 재생.
- `isaac/keyframes.py`, `isaac/test_keyframes.py` — 실기 고정 키프레임을 sim 관절각으로
  변환. 테스트 13개.

### 레포에 없는 것

**학습 env가 없다.** 논문의 Isaac Gym env는 공개되지 않았다. 즉 이 설계의 결과물은
재구현이지 복원이 아니다.

### 이미 측정된 사실 (2026-08-25, `--motion script --script-steps 6`)

```
t= 2.0s  base=(+0.094, +0.078, +0.101)   warmup 끝, 서 있음
t= 6.0s  base=(+0.173, +0.097, +0.097)   6걸음에 x +8cm 전진
t= 8.0s  base=(+0.246, +0.117, +0.051)   rise 시퀀스: 뻗고 굴러 넘어짐
t=26.0s  base=(+0.265, +0.116, +0.056)   그대로 누워 있음
```

- **걷기는 sim에서 동작한다** — 실기 키프레임 재생만으로 전진한다.
- **일어서기는 sim에서 실패한다** — `addRise()`가 눕히는 데까지는 가는데 못 일어난다.
  `joint1`/`joint3`가 관절 한계(∓0.87 rad)에 박힌 채 끝난다. 서보가 자기 체중을
  못 드는 상태다.

이 비대칭이 섹션 2의 출발점이다.

### 실행 환경

- Isaac Sim 5.1.0 / Isaac Lab 0.54.4, conda env **`env_isaaclab`**
  (`~/miniforge3/envs/env_isaaclab/bin/python`). 이 머신에서 isaaclab이 들어 있는
  파이썬은 이것뿐이다.
- rsl_rl / skrl / rl_games / stable-baselines3 모두 설치돼 있다. **rsl_rl을 쓴다** —
  Isaac Lab 기본이고 `export_policy_as_onnx()`
  (`isaaclab_rl/rsl_rl/exporter.py:25`)가 있다.
- 원격 렌더링은 `~/leisaac/STREAMING.md` 참조. `PUBLIC_IP=$(tailscale ip -4)` +
  `--livestream 1`이 필수다.

## 논문에서 확정된 사실

arXiv 2404.05932 본문에서 확인한 값이다. 추론이 아니라 명시된 수치다.

| 항목 | 값 |
|---|---|
| 시뮬레이터 / 알고리즘 | Isaac Gym + MJCF, PPO |
| 병렬 env | 131,072 |
| 제어 주기 | 10 Hz |
| 관측 | 40차원 = IMU 쿼터니언 + 서보 지령각, 4주기 이력 |
| 행동 | 6개 서보 지령각, ±50° |
| 학습량 | walk ≈ 100 epoch **+ 노이즈 추가 30 epoch**, stand ≈ 250 epoch |
| 하드웨어 | 156×156 mm, SG-90 마이크로서보 ×6, Arduino Nano Every, 총 ~$60 |

**walk 보상:** progress(`P−P_pre`) 30, height `min{1, |p|_z/0.08}` 20,
up `min{1, u_prj/0.93}` 5, heading 2, alive 1, death −1,
action `‖a−a_pre‖²` −2, vel −2

**walk 리셋:** 350 스텝 초과 / `‖q−[0,0,0,1]‖ > 0.7` / 좌면 모서리 접지 /
좌면 높이 < 5 mm

**stand 보상:** up 250, standing 100, spreading 50, death −1, action −2

**stand 리셋:** 350 스텝 / flip `u_prj < −0.7` /
fold `0.6 < u_prj 이고 max‖[θ₀,θ₁,θ₃,θ₅] − a_expand‖_∞ > 1`

**논문에 없는 것:** 보행 속도·성공률 등 정량 성능 지표가 전혀 없다. 정성 서술뿐이다
("RL 걸음새는 세 다리를 넓게 벌리고 몸통 진동으로 전진, 피치가 항상 위로 ~35° 사선
진행"). **따라서 1차 과업의 합격선은 우리가 정의해야 한다 — 섹션 7.**

## 섹션 1 — 패키지 배치와 태스크 등록

```
Chair-TypeAsymmetricalTripedalRobot/
├── src/                       (기존 ROS 실기 코드, 손대지 않음)
├── models/                    (기존 walk.onnx, stand.onnx — 비교 기준선)
├── mjcf/
├── docs/specs/                (이 문서)
└── isaac/
    ├── chair_sim.py           (기존 재생기)
    ├── keyframes.py           (기존)
    ├── pyproject.toml         ← 신규: pip install -e isaac/
    ├── chair_rl/              ← 신규 패키지
    │   ├── __init__.py            gym.register 2개
    │   ├── chair_asset.py         MJCF→USD 변환 + ArticulationCfg + 서보 액추에이터 모델
    │   ├── obs_layout.py          40차원 관측 레이아웃 규약 (실기와의 계약)
    │   ├── mdp.py                 순수 함수: 관측·보상·종료 수식 (torch만 의존)
    │   ├── base_env.py            두 태스크의 공통 DirectRLEnv
    │   ├── walk_env.py            보상·종료·초기상태만 다름
    │   ├── stand_env.py
    │   ├── events.py              도메인 랜덤화 EventTermCfg 묶음
    │   └── agents/rsl_rl_ppo_cfg.py
    ├── scripts/rsl_rl/{train.py,play.py}   IsaacLab 사본 + `import chair_rl` 한 줄
    ├── scripts/sysid_sweep.py     섹션 4의 파라미터 스윕
    └── tests/                     Isaac 없이 도는 단위 테스트
```

**태스크 ID:** `Chair-Walk-Direct-v0`, `Chair-Stand-Direct-v0`

**설치:** `pip install -e isaac/` (패키지명 `chair_rl`). 어느 디렉터리에서 실행하든
import가 풀리고, B로 갈아탈 때도 그대로다.

**왜 IsaacLab 트리가 아니라 이 레포인가.** `~/IsaacLab/source/isaaclab_tasks/`에 넣으면
upstream 업데이트와 얽히고 이 레포 git에 안 남는다. 논문 재현 코드는 논문 레포에 있어야 한다.

**왜 train.py를 복사하는가.** Isaac Lab의 `scripts/reinforcement_learning/rsl_rl/train.py`는
99행에서 `import isaaclab_tasks` 한 뒤 173행에서 `gym.make(--task)`를 부른다. 외부 패키지를
등록하려면 `gym.make` 전에 우리 패키지가 import돼야 해서, Isaac Lab 자체 external 템플릿도
train.py 사본에 그 import를 넣는 방식을 쓴다. **사본에는 어느 커밋에서 복사했는지 주석으로
남긴다** (upstream drift 대비).

**실행:**
```bash
pip install -e isaac/
python isaac/scripts/rsl_rl/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 4096
python isaac/scripts/rsl_rl/play.py  --task Chair-Walk-Direct-v0 --num_envs 16 --livestream 1
```

**기존 코드에 손대는 부분:** `chair_sim.py`의 `prepare_mjcf()`/`convert_mjcf()`를
`chair_rl/chair_asset.py`로 옮기고 `chair_sim.py`가 import하게 한다. 학습 env와 재생기가
**반드시 같은 USD·같은 액추에이터 파라미터**를 써야 한다. 여기가 갈리면 "학습에선 됐는데
재생기에선 안 되는" 유령을 쫓게 된다. `chair_sim.py`의 외부 동작은 유지한다.

## 섹션 2 — 시뮬 모델 정합성 (최대 위험 구간)

`mjcf/chair.xml`을 검토해 세 가지를 발견했다.

### ① 서보 질량이 모델에 없다

`servo1`~`servo6` geom(`density="1438"`)이 전부 주석 처리돼 있다. 구조물은
`density="175.5"`(경량 3D 프린팅)인데 SG90 9 g × 6 = 54 g가 통째로 빠졌다. 이 크기
로봇에서 무시 못 할 비중이고, 특히 다리 끝단 관성이 달라진다.

**원인:** 참조된 `main-SG90-MicroServo9g-TowerPro.1-*.STL` 파일이 레포에 없다.
그래서 주석 처리된 것으로 보인다.

**대응(현재 계획):** Isaac 쪽에서 질량만 더한다. `EventTermCfg`의
`randomize_rigid_body_mass`(operation="add")를 startup에 걸어 9 g × 6개를 부착 위치대로
얹는다. MJCF 주석 기준 servo2/3/6은 좌면(`chair` body), servo1/4/5는 각 브래킷 body.
`mjcf/chair.xml`을 건드리지 않아 MuJoCo 호환이 유지되고, 같은 term에 ±20% 산포를 주면
그대로 DR이 된다. **점질량 근사라 관성 텐서는 정확하지 않다 — 알려진 한계.**

**STL을 확보하면(경로 B):** 아래 "서보 STL 확보 시" 절 참조.

### ② 토크가 실제보다 1.7배 크다

```xml
<position name="joint1" ... forcerange="-0.3 0.3"/> <!--0.1764-->
```

`0.1764` N·m = 1.8 kgf·cm = SG90 실측 스톨 토크다. 저자가 실제값을 알면서 키워뒀다.
이 상태로 학습하면 **실기에서 못 드는 자세를 쓰는 정책**이 나온다.

### ③ 마찰 튜닝 이력이 전부 주석으로 죽어 있다

다리 geom에 `friction="0.1 0.0005 0.00001"`(아주 미끄럽게), floor에
`condim="1" friction="1 0.005 0.0001"` — 셋 다 주석. 지금 돌아가는 건 전부 기본값
모델이다. 비대칭 3족의 걸음새는 다리 미끄러짐에 의존할 공산이 크고, 이것이
`addRise()`가 sim에서 실패하는 **원인 후보 1번**이다.

### 그래서: 학습 전에 sim을 실측에 맞춘다 (system ID)

정답지가 이미 있다 — **실기는 이 키프레임으로 걷고 일어난다**(논문 영상). 그리고 그걸
재생하는 도구(`chair_sim.py --motion script`)가 있다. 이걸 계측 장비로 쓴다.

| 자유 파라미터 | 맞출 관측량 |
|---|---|
| 다리/바닥 마찰 μ | `addRise()` 성공 여부 ← **1순위** |
| effort_limit (0.1764 근방) | 한 걸음당 전진 거리·방향 |
| velocity_limit (SG90 0.1 s/60° ≈ 10 rad/s) | 서 있을 때 base 높이 |
| 액추에이터 지연 (`DelayedPDActuatorCfg`, min/max_delay) | 넘어지는 데 걸리는 시간 |
| 서보 질량 부착 방식, armature | |

`isaac/scripts/sysid_sweep.py`: 같은 키프레임 큐를 N개 병렬 env에 뿌리되 env마다 다른
파라미터를 주고 위 지표를 자동 기록한다. 헤드리스로 수백 조합이 한 번에 돈다.

> **섹션 2의 합격선: 키프레임 재생만으로 sim에서 일어서기가 성공해야 한다.**
> 이게 안 되면 학습 단계로 넘어가지 않는다. 모델이 틀린 채로 PPO를 돌리면 정책이 그 틀린
> 물리를 착취하고, 그건 실기에서 그대로 무너진다.

### Isaac Lab 매핑

`ImplicitActuatorCfg`(현재 `chair_sim.py`가 쓰는 것)에서 `DelayedPDActuatorCfg`로
갈아탄다. `min_delay`/`max_delay`를 물리스텝 단위로 받고 env별 랜덤화까지 되므로 서보
지령 지연 모델과 DR을 겸한다. `stiffness=40`(MJCF kp), `damping=0.01`,
`armature=0.001`, `effort_limit`/`velocity_limit`는 위 표대로.

### 서보 STL 확보 시

주석 처리된 geom에 `pos`도 `euler`도 없다 — 살아 있는 `leg1`/`bracket1`도 마찬가지다.
즉 **메시가 CAD 어셈블리 좌표계 그대로 export된 것**이고 배치가 STL에 구워져 있다.

- **경로 A — 저자 원본 export를 구한 경우:** `mjcf/mesh/`에 넣고 asset 6줄 + geom 6줄의
  주석을 풀면 끝. 위치·자세가 정확히 맞는다.
- **경로 B — 일반 SG90 모델:** 자기 로컬 좌표계에 있으므로 6개가 원점에 겹친다. MJCF에
  각 관절의 축과 위치가 명시돼 있으므로(예: `joint2 axis="0 1 0" pos=".13494 .06759 .10365"`)
  출력축을 joint axis에 정렬하고 joint pos에 앉히면 6 자유도 중 5개가 결정된다. 남는 축
  둘레 회전은 뷰어로 맞춘다.

어느 쪽이든:

- **질량은 `density` 대신 geom의 `mass="0.009"`로 못 박는다.** `density="1438"`은 저자
  메시 부피 기준으로 ~11 g가 나오게 고른 값이라 다른 STL을 쓰면 질량이 어긋난다.
- **충돌 처리를 정해야 한다.** 이 MJCF의 기본 geom은 `contype="0" conaffinity="1"`,
  바닥은 `contype="1" conaffinity="0"` — 로봇끼리는 충돌하지 않고 바닥과만 충돌한다.
  서보 geom을 살리면 **서보 몸통도 바닥에 닿기 시작한다.** 넘어진 자세의 접지점이 늘어
  일어서기 거동이 바뀐다. 실기에서도 닿는 게 맞으니 살리는 쪽이 옳지만, system ID를
  **변경 전후로 각각** 돌려 어느 쪽이 실기와 맞는지 확인한다.
- **검증:** 넣기 전후로 총질량·COM을 `--inspect`에 찍고, 키프레임 재생 지표(일어서기 성공,
  한 걸음 전진거리, 서 있을 때 base 높이)를 비교한다.

## 섹션 3 — 관측·행동 규약

### 40차원 레이아웃

`src/rl_walk.py` 실측 기준. 최신이 앞이다.

| 구간 | 내용 |
|---|---|
| `[0:16]` | 쿼터니언 이력 4×4, **(x, y, z, w)** 순, 최신이 index 0 |
| `[16:40]` | 액션 이력 4×6, 최신이 앞 |

**반드시 그대로 재현할 초기값 두 개:** 쿼터니언 이력은 `zeros`에 `w=1`,
**액션 이력은 `ones`** (`np.ones([4,6])`). 0이 아니다. 학습 때 이 초기 분포로 배웠기
때문에 배포 코드가 그렇게 돼 있다.

**갱신 순서:** 액션을 낸 **뒤에** 이력을 갱신한다. 즉 t 시점 액션은 t−1까지의 관측만 본다.
env의 `_get_observations`도 같은 순서여야 한다.

**IMU 부호 규약:** 실기는 `[-x, -y, z, w]`로 뒤집어 넣는다(장착 방향 보정). 학습 프레임을
"실기가 부호 반전을 마친 뒤의 프레임"으로 정의하고, sim에서는 Isaac의
`root_quat_w`(w,x,y,z)를 (x,y,z,w)로 재배열해 그대로 쓴다. `obs_layout.py`에 이 계약을
함수로 두고 테스트한다.

### 위치·자세의 기준 바디

논문의 `p`(높이)와 `u_prj`(직립도)는 모두 **좌면(seat)** 기준이다. Isaac에서는 MJCF의
`chair` 바디가 좌면에 해당한다. 루트 바디는 `dummy`(freejoint 부착점)이고 좌면과
오프셋이 있으므로, 보상·종료 계산에는 **`chair` 바디의 위치·자세를 쓴다.**
`chair_sim.py`의 로그가 쓰는 `root_pos_w`(= `dummy`)와 값이 다르다 — 섹션 2의
system ID 지표를 비교할 때 어느 쪽인지 항상 명시한다.

### 행동

절대 목표각(증분 아님), rad, `±0.872665`(=±50°)로 클립. 10 Hz 제어(물리 1/120 →
decimation 12).

### 미해결: 정책 출력 ↔ 물리 서보 대응

`chair_sim.py`의 `JOINT_ORDERS` 후보 3개 중 어느 것인지 레포만으로는 확정할 수 없다.
`embedded.ino`는 서보 인덱스를 핀 번호로만 매핑하고 관절 이름을 남기지 않는다.

학습은 우리가 고른 순서로 하면 되지만, **실기 이식 전에 서보를 하나씩 움직여 관절 대응을
실측으로 확정하는 작업이 반드시 필요하다.** 별도 과업으로 추적한다.

### ONNX 계약

입력 `[1, 40]`, 출력 이름 `mu`. 기존 `models/*.onnx`와 동일해야 `src/rl_walk.py`가
무수정으로 돈다. 내보낸 파일의 시그니처를 검사하는 테스트를 만든다.

## 섹션 4 — 보상·종료

논문 표를 `chair_rl/mdp.py`의 순수 함수로 옮기고, Direct env는 가중합만 한다.
각 term은 `extras["log"]`로 따로 찍어 텐서보드에서 개별 추적한다.

### walk 보상

| 논문 term | 구현 | 비고 |
|---|---|---|
| progress 30 | Ant 관례: `potential = −‖target − p‖/dt`, 보상 = `potential − potential_pre` | target은 +x 방향 먼 점. `src/utils.py`가 Ant `torch_jit_utils` 사본이라 이 관례가 맞다 |
| height 20 | `min(1, p_z / 0.08)` | 8 cm 이상 만점 |
| up 5 | `min(1, u_prj / 0.93)` | `u_prj`는 `src/utils.py`의 `compute_up_proj`와 동일 정의. 음수면 보상도 음수 |
| heading 2 | Ant 관례: `heading_proj` 정규화 후 클립 | **논문에 정규화 상수 없음** → Ant의 `/0.8` 채택 |
| alive 1 / death −1 | 상수 / 종료 시 | |
| action −2 | `‖a − a_pre‖²` | 크기가 아니라 **변화량** 페널티 |
| vel −2 | `‖q̇‖²` | **논문에 형태 없음** → L2 채택 |

### walk 종료

| 논문 조건 | 구현 | 짚을 점 |
|---|---|---|
| 350 스텝 초과 | **truncation** | termination과 구분한다. PPO가 시간 초과를 실패로 오해하면 가치함수가 망가진다. `_get_dones()`가 `(terminated, truncated)`를 나눠 반환하므로 정확히 매핑 |
| `‖q − [0,0,0,1]‖ > 0.7` | 쿼터니언 거리 | **회전각 약 82°에 해당**(‖q−q_id‖² = 2−2cos(θ/2), 0.49 대입). 그리고 **yaw도 센다** — 제자리에서 82° 돌면 리셋이다. 논문 RL 걸음새가 "35° 사선"으로 수렴한 게 이 종료조건의 압력일 가능성이 있다 |
| 좌면 모서리 접지 | `ContactSensor` | 우리 MJCF의 좌면은 geom 하나라 "모서리"가 별도 바디가 아니다. **1안(채택): 좌면 바디 접촉력 임계값** — 근사임을 명시. 2안: 모서리 4곳에 작은 충돌체 추가(충실, MJCF 수정) |
| 좌면 높이 < 5 mm | `p_z < 0.005` | |

`death −1`은 truncation이 아니라 termination에만 적용한다.

### stand 보상·종료

| 논문 term | 구현 |
|---|---|
| up 250 | `u_prj` |
| standing 100 | **공식 미명시** → `exp(−‖θ − θ_stand‖²/σ)`. `θ_stand`는 `keyframes.py`의 `STANDING_POS` 변환값 |
| spreading 50 | **공식 미명시** → `a_expand` 기준 근접도. `a_expand`는 `EXTENTION_POS` 변환값으로 **추론** |
| death −1, action −2 | walk과 동일 |

종료: 350 스텝(truncation) / flip `u_prj < −0.7` /
fold `0.6 < u_prj 이고 max‖[θ₀,θ₁,θ₃,θ₅] − a_expand‖_∞ > 1`.

**`keyframes.py`가 여기서 재사용된다.** `θ_stand`와 `a_expand`가 이미 만들어 둔 실기
키프레임 변환값이라, 학습 목표 자세와 실기 자세가 정의상 일치한다.

**성공 종료는 논문에 없다.** 학습 중엔 넣지 않고(보상으로 유도), 평가 하네스에서만
`u_prj > 0.95`(실기 FSM 임계값)를 성공 판정으로 쓴다. 학습 신호와 평가 지표를 분리해야
과적합을 못 본 채 넘어가지 않는다.

### 테스트

`mdp.py`가 torch만 의존하므로 Isaac 없이 전부 단위 테스트한다: 직립 쿼터니언에서
`up=1`, 90° 기울면 `up≈0`, 82° 근방에서 종료 경계가 켜지는지, truncation과 termination이
섞이지 않는지.

## 미작성 섹션 (5~8)

| # | 범위 |
|---|---|
| 5 | 초기 상태 분포 — 특히 stand의 "넘어진 자세" 샘플링을 어떻게 만들 것인가 |
| 6 | 도메인 랜덤화 목록 — 논문의 "노이즈 추가 30 epoch"에 해당하는 2단계 학습 포함 |
| 7 | 학습 배선(rsl_rl PPO 하이퍼파라미터, num_envs, 체크포인트)과 **1차 과업 합격선 정량화** |
| 8 | ONNX 내보내기와 기존 코드 접합(`src/rl_walk.py`, `chair_sim.py`) |

## 열린 질문 / 확정 필요

1. **좌면 모서리 접지를 접촉력 임계값으로 근사**하는 것 — 승인 대기.
2. **`a_expand = EXTENTION_POS` 추론** — fold 조건의 의미("거의 섰는데 다리가 안 펴짐")에서
   역추론한 것이라 확정 아님.
3. **정책 출력 ↔ 물리 서보 대응** — 실기 이식 전 실측 필요.
4. **서보 STL** — 확보 시 경로 A/B. 미확보 시 점질량으로 진행.
5. **논문 미명시 상수** — heading 정규화, vel 페널티 형태, standing/spreading 공식.
   섹션 7의 스윕 대상.
6. **1차 과업 합격선** — 논문에 성능 수치가 없으므로 우리가 정의해야 한다.

## 근거

- 논문: arXiv 2404.05932 본문 ("논문에서 확정된 사실"의 표는 본문에서 직접 인용).
- 코드: `src/rl_walk.py`(관측 구성·좌표 변환), `src/rl_stand.py`(FSM 임계값),
  `src/utils.py`(`compute_up_proj`), `src/config.py`(키프레임), `mjcf/chair.xml`.
- Isaac Lab 0.54.4 소스: `direct_rl_env.py:151`(events),
  `isaaclab_rl/rsl_rl/exporter.py:25`(ONNX), `rsl_rl/train.py:99,173`(등록 시점),
  `actuator_pd_cfg.py:52`(`DelayedPDActuatorCfg`), `manager_term_cfg.py:184`(history_length).
- 실측: 2026-08-25 `chair_sim.py --motion script` 헤드리스 로그.
