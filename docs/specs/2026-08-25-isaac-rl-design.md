# Isaac Sim 상에서 걷기·일어서기 정책 학습 — 설계

작성 2026-08-25. 갱신 2026-08-26.
상태: **섹션 1~9 작성 완료.** §4는 논문 표 실값으로, §2·§3은 2026-08-26 실측으로 정정됨.
§9(핵심 뼈대)가 구현 계획의 직접 입력이다.
섹션 2(system ID)가 나머지 전부의 선행조건이다 — 그것이 통과하기 전 학습은 시작하지 않는다.
**현재 §2는 미통과다**: 임포트 질량 오류(§2①)와 MJCF timestep 발산(§2⑤)을 먼저 고쳐야 한다.

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
  `joint1`/`joint3`가 관절 한계(∓0.87 rad)에 박힌 채 끝난다.

이 비대칭이 섹션 2의 출발점이다.

> **2026-08-26 정정 — 위 로그의 해석이 틀렸다.**
> "굴러 넘어짐 → 누워 있음"으로 읽었으나, 좌면 `u_prj`를 실제로 재보니 전 구간
> **0.88 아래로 내려간 적이 없다**(= 좌면이 30°도 안 기울었다). 로봇은 옆으로 누운
> 것이 아니라 **구르지 못하고 그 자리에 주저앉은** 것이다. 원인은 아래 ①의
> 임포트 질량 오류였고, 그것을 고치면 시퀀스대로 옆으로 눕는다(`u_prj → 0`).
> `base z`(= `dummy`) 값 자체는 재현된다 — 바뀐 것은 해석이다.

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
    │   ├── chair_asset.py         MJCF→USD 변환 + 후처리(질량 굽기) + ArticulationCfg  (§9.2)
    │   ├── mass_spec.py           바디별 질량·관성·COM 상수, 출처 MuJoCo  (§2①, §9.2)
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

`mjcf/chair.xml`과 **그 MJCF가 Isaac으로 변환된 결과**를 검토해 다섯 가지를 발견했다.
①과 ⑤는 2026-08-26 실측으로 추가된 것이고, ④는 같은 실측으로 **반증**됐다.

### ① 임포트된 질량이 전부 틀리다 — 12.5배 무겁다 ← **최대**

MuJoCo 3.8.1과 Isaac에 같은 `mjcf/chair.xml`을 물려 바디 질량을 대조했다. 저자가
MuJoCo 뷰어로 액추에이터를 손튜닝한 대상이 이 모델이므로 MuJoCo 값이 기준이다.

| 바디 | MuJoCo | Isaac 임포트 | 배수 |
|---|---|---|---|
| `dummy` (geom 없음) | **0.00 g** | **1000.00 g** | 폴백 질량 |
| `chair` (seat+face) | 122.87 g | 627.71 g | ×5.11 |
| bracket ×3 | 1.25 g | 10.66 g | ×8.53 |
| leg ×3 | 3.80 g | 21.79 g | ×5.73 |
| **합계** | **138.03 g** | **1725.06 g** | **×12.5** |

**기전은 특정됐다. Isaac MJCF 임포터는 `density="175.5"`를 무시하고 `1000 kg/m³ ×
볼록껍질 부피`를 쓴다.** 볼록껍질 부피를 직접 계산해 대조하면 bracket은 10.66 cc →
10.66 g로 네 자리까지, seat+face는 627.89 cc → 627.71 g로 0.03% 이내로 맞는다
(leg만 5% 낮은데, PhysX convex cooking의 정점 수 제한에 따른 단순화로 설명된다).
`dummy`는 geom이 하나도 없어 추론할 것이 없자 폴백 1 kg을 받았다.

**대응:** `chair_asset.py`의 USD 후처리 단계에서 질량과 관성을 **MassAPI로 덮어써 USD에
굽는다**(§9.2). 임포터가 밀도를 무시하는 이상 변환 직후의 USD를 믿으면 안 된다.
런타임 events가 아니라 파일에 굽는 이유는 재생기 `chair_sim.py`가 같은 파일을 쓰기
때문이다.
관성은 질량비 스케일이 아니라 MuJoCo의 `body_inertia`/`body_iquat`/`body_ipos`를
그대로 옮기는 것이 옳다 — 볼록껍질 기반 분포는 bracket에서 부피가 3.2배 어긋나 있어
질량만 맞춰도 관성 분포가 남는다.

**검증:** `--inspect`에 바디별 질량·총질량·COM을 찍고 MuJoCo 값과 대조하는 테스트를
둔다. 이 표가 회귀 테스트의 기준값이다.

### ② 서보 질량이 모델에 없다

`servo1`~`servo6` geom(`density="1438"`)이 전부 주석 처리돼 있다. 구조물은
`density="175.5"`(경량 3D 프린팅)인데 SG90 9 g × 6 = 54 g가 통째로 빠졌다.

**①이 확정된 뒤 이 항목의 비중이 크게 올라갔다.** 올바른 모델 질량이 138 g이므로
빠진 54 g은 **+39%**다(1725 g 기준으로 보면 3%처럼 보였다 — 그 인상은 임포트 오류가
만든 착시였다). 게다가 논문은 기립을 "다리를 휘둘러 만든 관성력"으로 설명하고 서보는
브래킷과 다리에 붙으므로, 그 동작이 의존하는 바로 그 관성을 바꾼다.

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

다만 ①을 고치고 나면 **이것은 병목이 아니다.** 138 g에서 다리 하나가 받는 하중은
약 0.45 N이고 지레팔 50 mm면 필요 토크가 0.023 N·m — `0.3`은 물론 실측값 `0.1764`도
한 자릿수 여유가 있다. 실기 충실도를 위해 `0.1764`로 낮추되, "토크가 모자라 못
일어난다"는 가설은 폐기한다.

### ③ 마찰 튜닝 이력이 전부 주석으로 죽어 있다

다리 geom에 `friction="0.1 0.0005 0.00001"`(아주 미끄럽게), floor에
`condim="1" friction="1 0.005 0.0001"` — 셋 다 주석. 지금 돌아가는 건 전부 기본값
모델이다.

> **2026-08-26 반증.** 이 항목은 "`addRise()`가 sim에서 실패하는 원인 후보 1번"으로
> 적혀 있었다. μ = 0.02 ~ 3.0을 쓸어 **세 가지 질량 조건(1725 g / 725 g / 138 g)에서
> 각각 재생했고, 기립 성공은 0회**다. 마찰이 실제로 먹고 있다는 것은 확인했다 —
> 재질값을 되읽어 검증했고, 최종 미끄러짐 거리가 μ에 따라 120 mm ~ 70 mm로 분명히
> 달라진다. 로봇이 *어디로* 미끄러지는지는 바꾸지만 *일어나는지*는 바꾸지 않는다.
>
> 마찰은 여전히 system ID 대상이지만(걸음새는 미끄러짐에 의존한다), **기립 실패의
> 원인 후보에서는 빠진다.**

### ⑤ MJCF에 적힌 timestep이 불안정하다

```xml
<option timestep="0.016" iterations="50" .../>
```

이 값에서 위치 액추에이터(`kp=40`)가 발산한다. MuJoCo에서 STANDING 자세를 유지하기만
해도 관절 오차가 **0.2012 rad(11.5°)** 까지 벌어지고 로봇이 0.7초 만에 넘어진다.

| timestep | 최대 관절오차 | 결과 |
|---|---|---|
| **0.016 (커밋된 값)** | 0.2012 rad | 서 있지도 못한다 |
| 0.008 | 0.0005 rad | 선다 |
| 0.004 이하 | 0.0003 rad | 선다 |

`integrator`를 `implicitfast`로 바꿔도 같다. **커밋된 MJCF는 자기 자신의 timestep에서
돌아가지 않는다.** Isaac 쪽은 `chair_sim.py`가 1/120을 쓰고 있어 이 문제를 겪지 않았고,
그래서 지금까지 드러나지 않았다.

**대응:** 물리 timestep은 **≤ 0.008** 로 못 박는다(현행 1/120 = 0.00833 유지). MJCF의
`0.016`은 어디서도 그대로 쓰지 않는다. MuJoCo로 교차검증할 때마다 이 값을 덮어써야
하므로, 비교용 MuJoCo 로더를 만든다면 그 안에 넣는다.

### 그래서: 학습 전에 sim을 실측에 맞춘다 (system ID)

정답지가 이미 있다 — **실기는 이 키프레임으로 걷고 일어난다**(논문 영상). 그리고 그걸
재생하는 도구(`chair_sim.py --motion script`)가 있다. 이걸 계측 장비로 쓴다.

**단, 자유 파라미터를 흔들기 전에 ①·⑤부터 고쳐야 한다.** 질량이 12.5배 틀리고
적분이 발산하는 모델 위에서 μ를 쓸어봐야 아무 의미가 없다 — 실제로 그렇게 해서
2026-08-26의 마찰 스윕 세 번이 전부 잘못된 질문에 답했다.

| 자유 파라미터 | 맞출 관측량 | 순위 |
|---|---|---|
| 서보 질량 54 g의 부착 위치·관성 | 기립 성공 여부 | **1** (138 g 대비 +39%) |
| 액추에이터 지연 (`DelayedPDActuatorCfg`, min/max_delay) | 기립 소요 시간, 넘어지는 데 걸리는 시간 | 2 |
| velocity_limit (SG90 0.1 s/60° ≈ 10 rad/s) | 스윙 속도 — 기립이 관성 의존이면 여기가 물린다 | 2 |
| 다리/바닥 마찰 μ | 한 걸음당 전진 거리·방향 | 3 (기립과는 **무관**, ④ 참조) |
| effort_limit | — | 낮음 (③ 참조: 한 자릿수 여유) |
| armature | 관절 떨림 | 낮음 |

`isaac/scripts/sysid_sweep.py`: 같은 키프레임 큐를 N개 병렬 env에 뿌리되 env마다 다른
파라미터를 주고 위 지표를 자동 기록한다. 헤드리스로 수백 조합이 한 번에 돈다.

### 섹션 2의 합격선 (2026-08-26 재정의)

> **루트가 수직인 웅크린 자세에서 시작해 `SLEEPING → STANDING` 구간을 재생하면
> 일어서야 한다.**
> 이게 안 되면 학습 단계로 넘어가지 않는다. 모델이 틀린 채로 PPO를 돌리면 정책이 그 틀린
> 물리를 착취하고, 그건 실기에서 그대로 무너진다.

**왜 다시 정의했나.** 이전 문구는 "키프레임 재생만으로 기립이 성공해야 한다"였는데,
명세가 덜 된 것이었다. `keyframes.build_rise()`(= `connect_performing.py`의 `addRise()`)는
`STANDING → EXTENTION → ROLLED → SLEEPING → STANDING`이라 **스스로 넘어지는 구간이 앞에
붙어 있다.** 그 앞부분은 데모용 장식이고 재현 대상이 아니다 —

- 논문 Table II의 기립은 `S₀ = 이미 오른쪽으로 누운 상태`에서 시작한다.
- Fig. 14는 사람이 손으로 넘어뜨린 뒤 일어나는 것이다.
- 안정한 timestep의 MuJoCo에서 `addRise()` 전체를 재생하면, 다리마찰 ≤ 1.0에서는 애초에
  **넘어지지도 않는다**(min `u_prj` = 0.672). 넘어지지 않는 것을 기립 실패로 세면 안 된다.

전체 시퀀스의 성공/실패를 세면 "넘어뜨리기가 잘 됐는가"와 "일어서기가 됐는가"가
뒤섞인다. 그래서 초기 자세를 고정하고 기립 구간만 잰다.

**2026-08-26 범위 축소:** 고정 키프레임은 임의 방향의 넘어진 자세를 복구하는 정책이
아니다. 따라서 이 게이트에서는 오른쪽·왼쪽·등 3종을 다루지 않는다. 루트 쿼터니언을
`(w,x,y,z)=(1,0,0,0)`으로 두고 관절을 `SLEEPING_POS`로 만든 뒤 물리적으로 안정화한다.
즉 검증 대상은 "다양한 각도에서 일어나기"가 아니라 **수직 상태에서 웅크렸다가 다리를
펴며 몸체를 들어 올리기**다. 다양한 초기 방향은 이후 학습 정책의 별도 과제다.

**판정:** `u_prj > 0.95`(실기 `src/rl_stand.py`의 FSM 임계값)이면서 articulation root 높이가 안정화
직후보다 10 mm 이상 올라가야 한다. 방향만 수직인 채 이미 낮게 서 있는 상태를 성공으로
잘못 세지 않도록 두 조건을 함께 본다. 2026-08-26 실측은 `u_prj 0.998 → 1.000`, root
높이 `0.061 → 0.101 m`(`+0.040 m`)로 통과했다.

### 수직 웅크림 기립을 직접 WebRTC로 재생하는 절차

현재 실행기는 아직 저장소에 승격되지 않은 진단용 스크래치다. 아래 절차는 이 문서를
작성한 워크스테이션에서 검증한 경로를 그대로 사용한다. `/tmp`가 정리되면
`rise_only.py`와 `keyframes.py`를 저장소의 `isaac/scripts/`로 옮긴 뒤 경로를 갱신해야
한다.

1. 기존 Isaac/WebRTC 실행을 종료한다. 같은 이름의 세션이 없다는 오류는 무시해도 된다.

   ```bash
   tmux kill-session -t crouch_rise_webrtc 2>/dev/null || true
   ```

2. 스크래치 디렉터리로 이동한다.

   ```bash
   cd /tmp/claude-1000/-home-tonnonssi-SOTA/1faa6ebb-76a4-40a5-8e39-8e75ce479186/scratchpad
   ```

3. Tailscale 주소를 WebRTC 서버에 넘기고 10회 재생한다. 매 회차는 웅크림 5초,
   `SLEEPING → STANDING` 보간 1.5초(20 Hz, 30프레임), 기립 유지 2.5초, 다음 회차 전
   1초 정지 순서다. `tmux`를 쓰는 이유는 터미널 연결이 끊겨도 Isaac 프로세스를 유지하기
   위해서다.

   ```bash
   STREAM_IP=$(tailscale ip -4)
   echo "$STREAM_IP"
   tmux new-session -d -s crouch_rise_webrtc \
     "PUBLIC_IP=$STREAM_IP /home/tonnonssi/miniforge3/envs/env_isaaclab/bin/python -u rise_only.py \
       --livestream 1 --loops 10 --settle 5.0 --hold 2.5 --pause 1.0 \
       2>&1 | tee crouch_rise_webrtc.log"
   ```

4. 준비 상태와 회차별 측정값을 확인한다.

   ```bash
   tmux capture-pane -pt crouch_rise_webrtc -S -80
   ```

   로그에 `준비 완료`가 나온 뒤 Isaac Sim WebRTC Streaming Client에서 3단계에 출력된
   Tailscale IPv4 주소로 접속한다. 화면이 이전 실행의 마지막 프레임에 머물면 클라이언트
   연결을 끊고 다시 연결한다. 각 회차의 `z`, `dz`, `u_prj`, `기립!` 판정은
   `crouch_rise_webrtc.log`에도 남는다.

5. 도중에 멈추거나 10회 종료 뒤 Isaac이 WebRTC 종료 처리에서 남아 있으면 세션을
   종료한다.

   ```bash
   tmux kill-session -t crouch_rise_webrtc
   ```

   종료 확인:

   ```bash
   pgrep -af 'rise_only.py.*livestream'
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```

   두 명령에서 `rise_only.py`나 해당 Isaac Python 프로세스가 나오지 않으면 종료된 것이다.

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

**타이밍 지연 (2026-08-27 기록).** 실기는 a_t 를 publish 한 직후 IMU 를 읽으므로 이력의
index 0 은 (a_t, a_t 작용 전 쿼터니언) 쌍이고, commands 큐(초기 SLEEPING/STANDING 2행 +
append/pop) 때문에 서보는 a_t 를 두 주기 뒤에 실행한다. env 는 (a_t, a_t 가 0.1 s 작용한
뒤 쿼터니언) 쌍이다. 한 주기 + 서보 두 주기의 차이는 학습 환경에서 고치지 않았고, 이슈
C(어느 쿼터니언을 짝지을지)와 §6(액션 지연 DR)에서 결정한다.

### 위치·자세의 기준 바디

논문의 `p`(좌면 중심)와 `u_prj`(직립도)는 **루트 바디(`dummy`)** 기준이다.

> **2026-08-26 정정.** 이전에는 "`chair` 바디를 쓴다"고 적었는데 반대다. MJCF의
> `dummy`는 `pos="0.095 0.0785 0.10365"`에 freejoint가 붙은 바디 — 좌면 크기의
> 절반과 좌면 높이다. **저자가 freejoint를 좌면 중심에 놓은 것**이고, Isaac Gym의
> root state = `dummy` = 논문의 `p`다. 실측이 이를 확인한다: 서 있을 때
> `dummy z = 0.101`, `chair z = −0.002`. `chair` 바디 원점은 메시 원점(바닥 높이의
> 모서리)이다.
>
> `chair`를 쓰면 height 보상 `min(1, p_z/0.08)`이 서 있어도 0이고, 리셋 조건
> "높이 < 5 mm"가 서 있는 채로 켜진다. 방향은 두 바디가 회전 없이 강결합이라
> 어느 쪽이든 같다.

따라서 보상·종료는 `root_pos_w`/`root_quat_w`를 그대로 쓴다. `chair_sim.py`의 로그와도
같은 값이다.

### 행동

절대 목표각(증분 아님), rad, `±0.872665`(=±50°)로 클립. 10 Hz 제어(물리 1/120 →
decimation 12).

### 미해결: 정책 출력 ↔ 물리 서보 대응

`chair_sim.py`의 `JOINT_ORDERS` 후보 3개 중 어느 것인지 레포만으로는 확정할 수 없다.
`embedded.ino`는 서보 인덱스를 핀 번호로만 매핑하고 관절 이름을 남기지 않는다.

학습은 우리가 고른 순서로 하면 되지만, **실기 이식 전에 서보를 하나씩 움직여 관절 대응을
실측으로 확정하는 작업이 반드시 필요하다.** 별도 과업으로 추적한다.

### ONNX 계약

입력 `[1, 40]`, 출력 이름 `mu`. 기존 `models/*.onnx`와 **그래프 계약이 동일**해야
`src/rl_walk.py`의 추론 호출부(`run_onnx_model()`)를 고치지 않고 쓸 수 있다.
다만 모델 경로는 실기 스크립트에 하드코딩돼 있어 이식 시 그 한 줄은 손대야 한다 —
계약의 범위는 그래프이지 파일 경로가 아니다. 상세는 섹션 8.

## 섹션 4 — 보상·종료

논문 표를 `chair_rl/mdp.py`의 순수 함수로 옮기고, Direct env는 가중합만 한다.
각 term은 `extras["log"]`로 따로 찍어 텐서보드에서 개별 추적한다.

### walk 보상

| 논문 term | 구현 | 비고 |
|---|---|---|
| progress 30 | `P = −‖p_target − p‖ / dt`, 보상 = `P − P_pre` | **`p_target = [10, 0, 0] m`** — 논문 명시. Ant `torch_jit_utils` 관례와 같다 |
| height 20 | `min(1, p_z / 0.08)` | 8 cm 이상 만점 |
| up 5 | `min(1, u_prj / 0.93)` | `u_prj = \|R_q e_z\|_z`. `src/utils.py`의 `compute_up_proj`와 동일 정의. 음수면 보상도 음수 |
| heading 2 | `min{1, (1/0.8)·R_q e_x · (p_target−p)/‖p_target−p‖}` | 논문 명시. `/0.8`은 Ant 관례와 **같은 값**이었다 |
| alive 1 / death −1 | 상수 / 종료 시 | |
| action −2 | `‖a − a_pre‖²` | 크기가 아니라 **변화량** 페널티 |
| vel −2 | `‖ω / (ω_max − ω_tol)‖²`, `ω_max = 10.472`, `ω_tol = 1` | 논문 명시. 단순 L2가 아니라 **정규화된** L2다. `10.472 rad/s = 600 °/s` ≈ SG90 무부하 속도 |

> 2026-08-26 정정: heading·vel·progress 세 항목은 "논문 미명시"로 적혀 있었으나 본문
> Table IV에 전부 수식으로 실려 있다. 위 표가 인용값이다.

### walk 종료

| 논문 조건 | 구현 | 짚을 점 |
|---|---|---|
| 350 스텝 초과 | **truncation** | termination과 구분한다. PPO가 시간 초과를 실패로 오해하면 가치함수가 망가진다. `_get_dones()`가 `(terminated, truncated)`를 나눠 반환하므로 정확히 매핑 |
| `‖q − [0,0,0,1]‖ > 0.7` | 쿼터니언 거리 | **회전각 약 82°에 해당**(‖q−q_id‖² = 2−2cos(θ/2), 0.49 대입). 그리고 **yaw도 센다** — 제자리에서 82° 돌면 리셋이다. 논문 RL 걸음새가 "35° 사선"으로 수렴한 게 이 종료조건의 압력일 가능성이 있다 |
| 좌면 모서리 접지 | 모서리 4개의 월드 z를 기하로 계산, `min < z_thresh` | 루트 = 좌면 중심이므로(§3) 모서리 = `root_pos + R_q·(±hx, ±hy, 0)`. 접촉 센서 없이 순수 함수로 계산되고 CPU 테스트가 된다. "접지"를 "z < 임계값"으로 근사하는 것은 같지만 근사가 한 줄로 명시된다. `hx, hy`는 좌면 메시 bounds에서 |
| 좌면 높이 < 5 mm | `p_z < 0.005` | |

`death −1`은 truncation이 아니라 termination에만 적용한다.

### stand 보상·종료

| 논문 term | 구현 |
|---|---|
| up 250 | `min{1, exp{2(u_prj − 1)}}` — **`u_prj` 자체가 아니다.** 직립 근방에서만 급히 커지는 지수형 |
| standing 100 | `1 / (2·\|arcsin(min{1, ‖a − a_stand‖/4})\| + 0.1)` (단 `u_prj > 0.85`, 아니면 0)<br>`a_stand = [−0.1745, 0, −0.1745, 0, 0.1745, 0]` |
| spreading 50 | `1 / (2·\|arcsin(min{1, ‖[θ₀,θ₁,θ₃,θ₅] − a_expand‖/4})\| + 0.1)` (단 `u_prj > 0.2`, 아니면 0)<br>`a_expand = [−1, −1, 1, −1]` |
| death −1, action −2 | walk과 동일 |

> 2026-08-26 정정: standing/spreading은 "공식 미명시"가 아니라 논문 Table VI에 실려 있다.
> **`a_expand`는 `EXTENTION_POS` 변환값이 아니라 `[−1, −1, 1, −1]` 리터럴이다** — 이전
> 추론은 폐기한다.

**세 가지 구현상 함의.**

1. **`a_stand`가 `keyframes.py`의 `STANDING_SIM`과 소수점까지 일치한다.**
   `[−0.1745, 0, −0.1745, 0, 0.1745, 0]` — 실기 `config.py`의 `STANDING_POS`를
   `simRad2realDeg()` 역변환한 값 그대로다. 논문의 액션 인덱스 규약이 이 레포의 변환과
   같다는 **독립 증거**이고, MJCF 트리 순회 순서(`JOINT_ORDERS["tree"]`)를 지지한다.
   섹션 3의 미해결 항목이 여기서 절반 닫힌다.

2. **`a_expand`는 관절 한계 밖이다.** `|±1| > 0.872665`이므로 도달 불가능한 목표다.
   즉 spreading 보상은 "다리를 한계까지 벌려라"로 작동한다. 최댓값에 못 닿는 것이
   버그가 아니라 설계다.

3. **보상 규모가 크게 비대칭이다.** arcsin 역수식은 `‖·‖ = 0`에서 `1/0.1 = 10`,
   `‖·‖ ≥ 4`에서 `1/(π + 0.1) ≈ 0.307`이다. 가중치를 곱하면 스텝당 최대
   standing 1000 / spreading 500 / up 250. **standing이 지배항**이고, `u_prj > 0.85`
   게이트가 켜지기 전에는 0이다. 이 계단이 학습 곡선에 그대로 보일 것이다 —
   term별 로깅(`extras["log"]`)이 여기서 필수다.

종료: 350 스텝(truncation) / flip `u_prj < −0.7` /
fold `0.6 < u_prj 이고 max‖[θ₀,θ₁,θ₃,θ₅] − a_expand‖_∞ > 1`.

**`keyframes.py`가 여기서 재사용된다.** `a_stand`가 `keyframes.py`의 `STANDING_SIM`과
같으므로 학습 목표 자세와 실기 자세가 정의상 일치한다. 다만 `a_expand`는 키프레임에서
오지 않는 별도 리터럴이므로 `mdp.py`에 상수로 못 박는다.

**성공 종료는 논문에 없다.** 학습 중엔 넣지 않고(보상으로 유도), 평가 하네스에서만
`u_prj > 0.95`(실기 FSM 임계값)를 성공 판정으로 쓴다. 학습 신호와 평가 지표를 분리해야
과적합을 못 본 채 넘어가지 않는다.

### 테스트

`mdp.py`가 torch만 의존하므로 Isaac 없이 전부 단위 테스트한다: 직립 쿼터니언에서
`up=1`, 90° 기울면 `up≈0`, 82° 근방에서 종료 경계가 켜지는지, truncation과 termination이
섞이지 않는지.

## 섹션 5 — 초기 상태 분포

에피소드 시작 상태는 학습의 절반이다. 특히 stand는 "어디서부터 일어나는가"가 곧 과제
정의다.

### walk

논문은 서 있는 자세에서 출발한다. 초기 관절각은 `a_stand`(= `keyframes.py`의
`STANDING_SIM`), base는 접지 상태.

**yaw는 랜덤화하지 않는다.** progress의 `p_target = [10, 0, 0]`과 heading이 +x를 고정
방향으로 박고 있어서, yaw를 흩뿌리면 "앞으로 걷기"가 "임의 방향으로 걷기"로 바뀐다.
같은 이유로 종료조건 `‖q − [0,0,0,1]‖ > 0.7`이 yaw도 세므로(≈82°) yaw 랜덤화는
시작하자마자 리셋되는 env를 만든다. 논문 재현이 목적이면 0 고정이 맞다.

랜덤화는 좁게만: 관절각 `a_stand ± 0.02 rad`, base 높이 `±2 mm`, roll/pitch `±0.02 rad`.
목적은 다양성이 아니라 **결정론적 초기값이 만드는 degenerate 정책 방지**다.

### stand — 넘어진 자세 3종

논문 명시: "right side, left side, and back on the ground" 세 자세에서 동시에 학습한다.
커리큘럼이 아니라 **동시**다. `env_id % 3`으로 균등 배분한다.

자세를 만드는 방법은 세 가지가 있다.

| 안 | 방법 | 문제 |
|---|---|---|
| A | base 쿼터니언을 손으로 지정 (오른쪽 = roll +90°, 왼쪽 = roll −90°, 등 = pitch −90°) | 결정적·재현 가능. 다만 지정한 자세가 물리적으로 안정한 접지 상태라는 보장이 없다 |
| B | 공중에서 떨어뜨려 안정화(drop & settle) | 자연스럽지만 분포가 물리에 의존해 섹션 2 전에는 정할 수 없고, 재현도 어렵다 |
| C | 실기 로그 재생 | 실기를 못 돌리는 현재 불가 |

**A + settle을 채택한다.** A로 자세를 놓고, 액션 없이 20 물리스텝(≈0.17 s) 동안
물리만 돌려 관통·떨림을 가라앉힌 뒤 에피소드 스텝 0을 시작한다. settle 구간은 보상도
종료 판정도 하지 않는다.

초기 관절각은 `a_stand`로 두고 settle에 맡긴다 — 논문이 이 값을 밝히지 않았고,
`SLEEPING_POS` 변환값을 쓰면 키프레임 기립 시퀀스의 사전지식을 학습에 주입하게 된다.
재현 목적에 맞지 않는다.

**시작 시 flip 리셋에 걸리지 않는지 확인해야 한다.** 세 자세 모두 `u_prj ≈ 0`이라
`u_prj < −0.7` 조건에는 안전하다. fold 조건(`0.6 < u_prj`)도 시작 시엔 꺼져 있다.
단위 테스트로 못 박는다.

### 히스토리 버퍼 초기화

섹션 3대로 쿼터니언 이력은 단위 쿼터니언, 액션 이력은 `ones`다.

여기에 의도적인 부정합이 있다. **stand는 넘어진 자세에서 시작하는데 쿼터니언 이력은
"직립"으로 채워진다.** 즉 정책은 첫 4스텝 동안 "방금 넘어졌다"고 본다. 실기
`src/rl_stand.py`도 정확히 이렇게 동작하므로 **고치지 않고 그대로 재현한다.** 고치면
실기와 관측 분포가 어긋난다.

## 섹션 6 — 도메인 랜덤화와 2단계 학습

논문: 걷기 1단계 학습 후 **"물리 파라미터·센서값·액션값에 무작위 노이즈를 추가해"**
약 30 epoch를 더 학습했다. 세 범주가 본문에 명시돼 있다.

### 왜 2단계인가

논문이 그렇게 했고, 이 로봇은 토크 여유가 거의 없다(섹션 2 ②). DR을 처음부터 켜면
정책이 아예 안 붙을 위험이 있다. 단계를 나누면 "노이즈 때문에 깨졌다"를 **분리해서
관측**할 수 있다 — 1단계 체크포인트가 대조군으로 남는다.

2단계는 1단계 체크포인트에서 이어서 학습한다. 처음부터 다시 돌리지 않는다.

### 항목

| 범주 | 항목 | Isaac Lab 수단 | 범위(초안) |
|---|---|---|---|
| 물리 | 다리·바닥 마찰 | `randomize_rigid_body_material` (startup) | §2 SysID 값 ±30% |
| 물리 | 서보 질량 | `randomize_rigid_body_mass` (add) | 9 g ±20% |
| 물리 | stiffness / damping | `randomize_actuator_gains` | ±20% |
| 물리 | effort / velocity limit | 액추에이터 cfg | ±20% |
| 물리 | 액추에이터 지연 | `DelayedPDActuatorCfg` `min/max_delay` | §2 SysID 값 ±1 물리스텝, env별 |
| 센서 | 쿼터니언 노이즈 | 관측에 가산 후 재정규화 | 각도 σ ≈ 1~2° |
| 액션 | 지령각 노이즈 | 관절에 보내는 값에 가산 | σ ≈ 0.5~1° (서보 백래시) |

범위는 초안이다. 마찰은 섹션 2가 값을 확정한 **뒤에** ±를 잡는다 — 중심값을 모르는 채
산포부터 정하는 건 순서가 거꾸로다.

### 반드시 지킬 두 가지 배선

**① 액션 노이즈는 관측 이력에 들어가면 안 된다.**
실기는 정책이 낸 값을 그대로 이력에 넣는다(`src/rl_walk.py`). 따라서 sim도
**노이즈 전 값을 이력에 기록**하고 **노이즈 후 값을 관절에 보낸다.** 뒤집으면 정책이
실기에는 존재하지 않는 정보(실제로 관절에 간 값)를 학습하게 되고, 실기에서 조용히
성능이 빠진다.

**② 센서 노이즈는 관측에만, 보상·종료에는 참값.**
실기 IMU는 이미 노이즈 낀 값을 주므로 관측에 노이즈를 넣는 게 옳다. 그러나 보상과
종료 판정은 학습 신호이지 로봇이 보는 값이 아니다. 노이즈 낀 `u_prj`로 종료를
판정하면 에피소드가 무작위로 끊긴다.

이 두 배선은 `mdp.py` 바깥(env 클래스)에서 일어나므로 순수 함수 테스트로 못 잡는다.
**env 레벨 테스트를 따로 둔다** — 액션에 큰 노이즈를 강제로 주입하고 다음 스텝 관측의
액션 구간이 노이즈 전 값과 같은지 검사.

## 섹션 7 — 학습 배선과 1차 과업 합격선

### 규모 환산 — epoch을 옮기지 않고 env-step으로 옮긴다

논문은 **131,072 env**다. 4090 한 장에 올라가지 않는다. 그리고 논문은
`horizon_length`를 밝히지 않았다. 따라서 "100 epoch"을 그대로 쓰면 안 된다 —
epoch당 샘플 수가 두 자릿수 배 다르다.

총 env-step으로 환산한다.

| | 논문 epoch | horizon 16 가정 | horizon 32 가정 |
|---|---|---|---|
| walk 1단계 | 100 | 210 M | 419 M |
| walk 2단계(노이즈) | 30 | 63 M | 126 M |
| stand | 250 | 524 M | 1.05 G |

**`horizon_length`는 가정이며 논문에 없다.** 그래서 목표를 점이 아니라 구간으로 잡는다:
walk 200 M ± 2배, stand 500 M ± 2배. 판정은 숫자 도달이 아니라 **보상 곡선이 평평해지고
합격선을 넘는지**로 한다.

**벽시계 시간은 여기에 적지 않는다.** M0(처리량 측정) 전에 시간을 말하면 그건 지어낸
숫자다. `num_envs` 1024 / 4096 / 8192 / 16384에서 steps/s를 재고 나서 채운다.

### 하이퍼파라미터

네트워크 구조는 **가정이 아니라 측정값**이다. `models/*.onnx`가 증명한다:

- MLP `[1024, 512]`, 활성함수 **ELU** (`actor_mlp.0.weight [1024, 40]`, `.2.weight [512, 1024]`)
- **상태무관 sigma** (`sigma [6]` 텐서 존재 → `log_std`가 관측의 함수가 아니다)
- **관측 정규화 켬** (RunningMeanStd가 그래프에 융합돼 있다)

이 셋은 그대로 쓴다. 나머지는 논문 미공개이므로 IsaacGymEnvs Ant/Humanoid 기본값에서
출발한다(§4에서 확인했듯 보상 설계가 그 계보다): `lr 3e-4` adaptive,
`gamma 0.99`, `lam 0.95`, `clip 0.2`, `entropy 0.0`.

### 1차 과업 합격선

논문에 정량 성능 지표가 없다. 그래서 **두 축**으로 정의한다.

**축 1 — 절대 기준**

| | 지표 | 합격선 |
|---|---|---|
| walk | 35 s 에피소드 완주(넘어짐 0) | 20 에피소드 중 ≥ 18 |
| walk | 평균 전진 속도 | ≥ 0.02 m/s |
| walk | 좌면 평균 높이 | ≥ 0.08 m (height 만점 기준) |
| stand | 3종 자세별 기립 성공(`u_prj > 0.95` 도달) | 각 20회 중 ≥ 16 |
| stand | 기립 소요 시간 | ≤ 1.5 s |

`0.02 m/s`는 논문 Fig.11에서 10 s에 x ≈ +0.25 m를 읽은 값이고, `1.5 s`는 Fig.13의
"Using learned model" 구간(≈1.4 s)에서 왔다. **둘 다 그래프에서 눈으로 읽은 값이라
정밀하지 않다.** 그래서 축 2가 필요하다.

**축 2 — 같은 시뮬 위의 논문 정책 (실질 기준선)**

`models/walk.onnx`·`stand.onnx`를 **우리 평가 하네스로, 우리 시뮬에서** 돌린 값을
기준선으로 삼는다. 우리 정책이 이보다 나쁘면 재현 실패다.

이게 핵심이다. 논문 그림과 비교하면 "시뮬이 다른 것"과 "정책이 나쁜 것"이 섞인다.
같은 시뮬 위에서 비교하면 시뮬 차이가 상쇄되고 **정책 품질만 남는다.** 논문 정책이
우리 시뮬에서 잘 못 걷는다면 그건 우리 정책이 아니라 섹션 2가 아직 덜 끝났다는
신호이고, 그 사실 자체가 유용한 진단이다.

### 평가 하네스

`isaac/scripts/eval.py` — 정책(체크포인트 또는 ONNX)을 받아 위 지표를 JSON으로 뱉는다.
학습 코드와 분리한다. 그래야 논문 정책과 우리 정책에 **같은 자를 댈 수 있다.**

`play.py`는 여기에 더해 논문 Fig.10~13에 대응하는 플롯을 그린다: 10 s 구간의 x/y,
roll/pitch/yaw, 서보 지령각 6개. 정성 비교("세 다리를 벌리고 몸통 진동으로 전진,
피치가 항상 위로 ~35°")는 이 그림으로만 판정할 수 있다.

## 섹션 8 — ONNX 내보내기와 기존 코드 접합

### 계약

`src/rl_walk.py:53`이 `session.run(["mu"], {input_name: obs})`로 **출력 이름을
하드코딩**한다. 그리고 관측 정규화가 그래프 안에 있어야 한다 — 실기 코드에는 정규화
단계가 없다. 따라서 내보낸 파일은 다음을 만족해야 한다.

- 입력 `obs [1, 40]`, 출력 `mu [1, 6]`
- 정규화(`Sub`/`Div`/`Clip`)가 그래프 **선두에 융합**
- 결정론적 출력 (평가 시 `mu`만, 샘플링 없음)

### rsl_rl 유지 + 어댑터 (권고)

섹션 1은 rsl_rl을 택했다. 그런데 `models/*.onnx`는 rl_games 산출물이다
(`model._model.a2c_network.*`). rsl_rl의 `export_policy_as_onnx()`는 그래프 모양과
출력 이름이 다르므로 계약을 자동으로 만족하지 않는다.

두 갈래가 있다.

| 안 | 내용 | 대가 |
|---|---|---|
| A | rl_games로 전환 | 계약이 공짜로 충족되고 논문과 같은 스택. 섹션 1 결정을 뒤집고 Isaac Lab 기본 경로에서 벗어난다 |
| B | rsl_rl 유지 + 내보내기 어댑터 | `nn.Module` 래퍼 하나(정규화 레이어를 앞에 붙이고 출력 이름을 `mu`로 지정) → `torch.onnx.export`. 50줄 남짓 |

**B를 권고한다.** 섹션 1이 rsl_rl을 고른 이유(Isaac Lab 기본, 유지보수, `events` API
공유)는 여전히 유효하다. 우리가 맞춰야 하는 것은 **ONNX 그래프 계약**이지 학습
라이브러리가 아니다. 어댑터가 커지거나 rsl_rl의 정규화 구현이 rl_games와 수치적으로
어긋나면 그때 A로 되돌린다 — 되돌리는 비용이 작다.

### 계약 테스트 (`tests/test_onnx_contract.py`, Isaac 불필요)

1. 내보낸 파일의 입출력 이름·shape가 `models/walk.onnx`와 동일한가
2. 같은 관측 100개에 대해 torch 정책 출력과 onnxruntime 출력이 `1e-5` 이내인가
3. `src/rl_walk.py`의 `run_onnx_model()`을 **그대로 import해** 호출이 성공하는가

3번이 핵심이다. 실기 코드를 복사해 흉내 내면 계약이 아니라 우리 해석을 테스트하게 된다.

### 기존 코드 접합

**`models/`를 덮어쓰지 않는다.** `models/reproduced/{walk,stand}.onnx`로 내보낸다.
논문 정책은 섹션 7 축 2의 기준선이다 — 지우면 비교 대상을 잃는다.

`chair_sim.py`에 `--policy PATH`를 추가해 임의 경로를 받게 한다. 기본값은 현재대로
`models/{motion}.onnx`이므로 기존 동작은 그대로다.

실기 스크립트(`src/rl_walk.py`, `src/rl_stand.py`)는 모델 경로가 하드코딩돼 있다.
**이 단계에서는 건드리지 않는다** — 실기 이식은 별도 과업이고, 그때 경로를 인자로
받도록 함께 고친다.

## 섹션 9 — 핵심 뼈대: 단위·에셋·데이터 흐름

2026-08-26 확정. 첫 태스크는 **걷기**다 — 키프레임 재생으로 sim에서 이미 되고(§2 실측),
§2 게이트(넘어진 좌면 뒤집기)와 무관하다. 그래서 뼈대를 §2 미통과 상태에서도 끝까지
검증할 수 있다. `stand_env`는 §2 통과 후 별도 이슈다.

### 9.1 단위와 경계

| 단위 | 하는 일 | 의존 | 검증 |
|---|---|---|---|
| `chair_asset.py` | MJCF → USD 빌드 + 후처리, `ArticulationCfg` 제공 | Isaac | 빌드된 USD의 질량·COM을 MuJoCo 값과 대조 |
| `mass_spec.py` | 바디별 질량·관성·COM 상수 (출처 MuJoCo) + 서보 옵션 | 없음 | mujoco로 `chair.xml`을 읽은 값과 일치 |
| `obs_layout.py` | 40차원 레이아웃 상수, 이력 버퍼 push/reset | torch만 | `src/rl_walk.py`의 numpy 로직과 동일 출력 |
| `mdp.py` | 보상 항·종료 조건 순수 함수 (Table III/IV) | torch만 | 해석적 케이스 |
| `base_env.py` | `DirectRLEnv` 훅 구현. 위 넷을 부르는 얇은 껍데기 | 위 넷 + Isaac | 16 env 스모크 |
| `walk_env.py` | `WalkEnvCfg` + 가중합·종료 조합 | base_env | 스모크에 포함 |

**규칙: `obs_layout`·`mdp`·`mass_spec`은 `self`를 모른다.** 텐서를 받아 텐서를 돌려준다.
Isaac 없이 CPU에서 pytest가 돈다. §1이 약속한 "Manager-based로 갈아탈 수 있는 구조"의
실체가 이것이다.

### 9.2 에셋 빌드 — USD 하나가 진실이다

```
mjcf/chair.xml
  → prepare_mjcf():  floor/light 제거 + 무명 <body>에 이름 부여 (bracket1..3, leg1..3)
  → MjcfConverter:   USD 생성
  → postprocess():   ① worldBody의 여분 ArticulationRootAPI 제거 (USD에 저장)
                     ② 바디 8개에 MassAPI 덮어쓰기 — mass / diagonalInertia /
                        principalAxes / centerOfMass  ← 값의 출처 = mass_spec.py
  → isaac/usd/chair_<spec-hash>.usd   (스펙이 바뀌면 파일명이 바뀐다 = 캐시 무효화)
```

**질량 교정을 런타임 events가 아니라 USD에 굽는다.** §1이 "학습 env와 재생기는 반드시
같은 USD"라고 못 박았다. events로 하면 `chair_sim.py`는 여전히 1725 g 모델을 재생한다.
USD에 구우면 둘 다 공짜로 맞는다.

**관성은 질량비 스케일이 아니라 직접 이식한다.** MuJoCo의 `body_inertia`(주축 관성)·
`body_iquat`(주축 방향)·`body_ipos`(COM)가 USD `MassAPI`의 `diagonalInertia`·
`principalAxes`·`centerOfMass`와 **정확히 같은 세 필드**다. 변환이 없다. 남는 전제는
"임포터가 MJCF 바디 프레임을 USD 프림 프레임으로 보존한다"이고, 빌드 테스트의
`get_coms()` 되읽기가 판정한다(열린 질문 #2).

**무명 바디에 이름을 준다.** 현재 `_body_0`~`_body_5`는 임포터가 붙인 이름이라 순서
보장이 없다. MJCF 전처리에서 `bracket1`/`leg1`…을 박으면 질량 스펙·관절 매핑·로그가
전부 이름으로 돈다. `chair`·`dummy`가 이름을 유지한 채 넘어온 것으로 임포터가 이름을
존중함은 확인됐다.

**서보 54 g(§2②)은 `MassSpec`의 옵션 필드다.** 켜면 해시가 바뀌어 별도 USD가 나온다.
기본값은 **MuJoCo 그대로(꺼짐)** — 저자 모델이 기준선이고, 서보 추가는 §2 system ID가
판정할 실험 변수다.

`chair_sim.py`는 `chair_asset.build_usd()`를 import해 쓰도록 바꾼다. 외부 동작(플래그,
출력)은 유지하되 재생되는 모델의 질량이 138 g으로 바뀐다 — 이것은 수정이지 회귀가 아니다.

### 9.3 한 스텝의 데이터 흐름

`DirectRLEnv.step()`의 실제 호출 순서(소스 확인)에 훅을 맞춘다. 순서가 곧 정확성이다.

```
_pre_physics_step(a)   a: (N,6) 정책 출력(raw) → 이력용 self._raw_act; clip ±0.8727 → 관절용 self._act
                       ※ 액션 노이즈(§6)는 관절로 보내는 사본에만. 이력에는 원본이 들어간다
  ×12 { _apply_action  robot.set_joint_position_target(self._act[:, a2j])
        sim.step }                                   ↑ 정책 인덱스 → 관절 인덱스 (find_joints, preserve_order)
_get_dones()           mdp.walk_terminated(root_pos, root_quat) ,
                       episode_length_buf >= 350        → (terminated, truncated)
_get_rewards()         mdp.walk_rewards(...) 7항 → cfg 가중치로 합. death는 terminated에만.
                       각 항을 extras["log"]에 개별 기록
_reset_idx(ids)        scene.reset → root/joint 초기화(§5) → potentials 재계산
                       → obs_layout.reset_history(rot_his, act_his, ids)      ← 반드시 여기
_get_observations()    obs_layout.push(rot_his, act_his, root_quat_xyzw, raw_action, skip_mask=reset_buf)
                       → (N,40) = [rot 4×4 | act 4×6], 최신이 앞
```

**리셋 직후 첫 관측 (2026-08-27 정정).** 관측은 리셋 뒤에 계산되므로 push 를 그대로
하면 첫 관측에 실측 쿼터니언이 들어간다. 실기는 초기 이력으로 첫 추론을 하므로,
리셋된 env 는 `skip_mask=reset_buf` 로 push 를 건너뛰어 첫 관측을 리셋값 그대로 둔다.
§9.6 스모크 테스트의 "리셋 직후 obs 이력 = 규약값" 은 이 규칙으로 성립한다.

**이력의 액션은 클립 전 (2026-08-27 정정).** 실기 rl_walk.py 는 ONNX 출력 mu 를
그대로 action_history 에 넣고 safeClip 은 서보 지령에만 건다. env 도 같다: 이력 = raw,
관절 목표 = clip. ACT_INIT=1.0 이 관절 한계 밖인 것이 그 정황이다.

**이력 리셋이 `_reset_idx` 안에 있어야 하는 이유:** 관측은 리셋 **뒤에** 계산된다.
다른 곳에서 하면 리셋된 env의 첫 관측에 죽은 에피소드의 이력이 섞인다. 리셋값은 §3
그대로 — 쿼터니언 `(0,0,0,1)`, 액션 **`1.0`**.

**갱신 순서:** `push`는 "이번 스텝 액션을 낸 뒤" 이력에 넣는다. `src/rl_walk.py`와
같다. `_get_observations`가 `_pre_physics_step` 다음에 불리므로 자연히 맞는다.

**progress 항의 상태:** `potentials (N,)`를 env가 들고, 리셋 시 `−‖p_target − p‖/dt`로
재초기화한다. `mdp.progress(potentials, root_pos, p_target, dt) → (reward, new_potentials)`
순수 함수.

**관절 순서:** 아티큘레이션이 보고하는 관절 순서는 `[joint2, joint4, joint6, joint1,
joint3, joint5]`(실측, 임포터의 폭우선 순회)이고 정책 인덱스 순서는 §3의
`[joint2, joint1, joint4, joint3, joint6, joint5]`다. 둘이 다르므로 `a2j` 인덱스
텐서가 필수이며, `obs_layout.POLICY_JOINT_NAMES` 상수 하나에서만 나온다.

### 9.4 종료 조건 — 접촉 센서 없이 전부 순수 함수

루트 = 좌면 중심(§3)이므로 모서리 4개의 월드 z를 기하로 계산한다:

```
corner_local (4,3) = (±hx, ±hy, 0)                       # 좌면 반폭, 메시 bounds에서
corner_z (N,4)     = (quat_rotate(root_quat, corner_local) + root_pos)[..., 2]
ground             = corner_z.min(-1) < z_thresh
```

결정적이고, ContactSensor 설정·필터 문제가 없고, CPU에서 테스트된다. 네 조건(tilt·
ground·height·max_episode)이 `mdp.walk_terminated()` 하나로 들어간다.

### 9.5 Cfg와 등록

```python
@configclass
class WalkEnvCfg(DirectRLEnvCfg):
    sim = SimulationCfg(dt=1/120, render_interval=12)   # §2⑤: ≤ 0.008
    decimation = 12                                     # → 10 Hz
    episode_length_s = 35.0                             # → 350 스텝
    observation_space = 40;  action_space = 6
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=0.6)
    robot = chair_asset.articulation_cfg(mass_spec=MUJOCO)   # /World/envs/env_.*/Robot
    # 보상 가중치·리셋 임계값은 cfg 필드. 기본값 = Table III/IV
```

`gym.register("Chair-Walk-Direct-v0")`에 **rl_games와 rsl_rl cfg entry point를 둘 다**
건다(`isaaclab_tasks`의 cartpole 관례). 뼈대는 라이브러리를 모르고, 선택(§8)은
`train.py` 시점으로 미룬다.

### 9.6 테스트

| 테스트 | Isaac | 검증 내용 |
|---|---|---|
| `test_obs_layout.py` | 불필요 | `src/rl_walk.py`의 이력 갱신을 **그대로 import**해 100스텝 무작위 입력에 동일 출력 |
| `test_mdp.py` | 불필요 | 직립→`up=1`, 90°→`up≈0`, 82°에서 tilt 경계, truncation≠termination, death는 terminated에만 |
| `test_mass_spec.py` | 불필요 (mujoco) | 스펙 상수 = MuJoCo가 `chair.xml`에서 계산한 값. mujoco는 `lerobot` env에만 있으므로 스펙 갱신 시 수동 실행, CI 밖 |
| `test_asset_build.py` | 필요 | 빌드된 USD 되읽기: 총질량 138.03 g, 바디별 질량·COM, 관절 6개·이름, 아티큘레이션 루트 1개 |
| `test_env_smoke.py` | 필요 | 16 env × 50 스텝: obs `(16,40)` NaN 없음, 리셋 직후 obs 이력 = 규약값, `a2j`로 관절이 움직임 |

Isaac 필요 테스트는 `pytest -m isaac`으로 분리한다.

### 9.7 이슈 분할 (각 ≤ 400줄)

| | 내용 | 규모 |
|---|---|---|
| A | `chair_asset.py` + `mass_spec.py` + 빌드 테스트. `chair_sim.py`가 이걸 import하도록 | M |
| B | `obs_layout.py` + `mdp.py` + CPU 테스트 | M |
| C | `base_env.py` + `walk_env.py` + 등록 + 스모크 | M |

A는 C 없이도 가치가 있다 — `chair_sim.py`가 즉시 올바른 질량으로 재생한다.

## 열린 질문 / 확정 필요

**열려 있는 것**

1. **서보 STL** — 확보 시 경로 A/B. 미확보 시 점질량으로 진행. §2②에서 우선순위가
   1순위로 올라갔으므로, 점질량 근사로 충분한지를 §2 게이트로 판정해야 한다.
2. **USD 임포트가 MJCF 바디 프레임을 보존하는가** — §9.2의 MassAPI 이식이 이것을
   전제한다. 빌드 테스트의 `get_coms()` 되읽기로 판정.
3. **rsl_rl vs rl_games** — 섹션 8. §9.5가 두 entry point를 동시에 등록하므로
   `train.py` 시점까지 미룰 수 있다. `models/*.onnx`가 rl_games 산출물임이 확인됐다.
   현재 권고는 rsl_rl 유지 + 내보내기 어댑터.
4. **`horizon_length` 미상** — 논문이 밝히지 않아 학습량 환산에 가정이 들어간다. 섹션 7.

**닫힌 것 (2026-08-26, 논문 본문 재확인)**

| 이전 항목 | 결과 |
|---|---|
| `a_expand = EXTENTION_POS` 추론 | **반증.** Table VI에 `[−1, −1, 1, −1]` 리터럴로 명시 |
| 논문 미명시 상수(heading·vel·standing·spreading) | **전부 명시돼 있었다.** Table IV/VI 인용으로 §4 교체 |
| 1차 과업 합격선 | 섹션 7에서 정의 (논문 정책을 **같은 시뮬 위에서** 돌린 값이 기준선) |
| 마찰이 `addRise()` 실패의 원인 후보 1번 | **반증.** μ=0.02~3.0 × 질량 3조건에서 기립 0회 (§2④) |
| 토크 부족이 기립 실패의 원인 | **반증.** 138 g에서 필요 토크 0.023 N·m, 한 자릿수 여유 (§2③). 수직 웅크림에서 `SLEEPING→STANDING` 재생 시 좌면 z 0.061→0.101 m, 10/10 기립 — 서보는 몸통을 든다 |
| 좌면 모서리 접지를 접촉력으로 근사 | **대체.** 루트 = 좌면 중심이므로 모서리 z를 기하로 계산 (§4, §9.4) |
| 관성을 질량비 스케일로 근사 | **대체.** MuJoCo `body_inertia/iquat/ipos` → MassAPI `diagonalInertia/principalAxes/centerOfMass` 직접 이식 (§9.2) |
| `chair` 바디가 좌면 기준 | **반증.** `dummy`(루트)가 좌면 중심이다 (§3) |

**범위가 줄어든 것**

- **정책 출력 ↔ 물리 서보 대응.** `a_stand ≡ STANDING_SIM` 일치로 *액션 인덱스 규약*은
  확정됐고 `JOINT_ORDERS["tree"]`가 유력하다. 남은 것은 *실기 서보 번호 ↔ 관절 이름*
  대응뿐이며, 이것만 실기 이식 전 실측이 필요하다.

## 근거

- 논문: arXiv 2404.05932 본문. §4의 보상·종료 수식은 Table III~VI 직접 인용
  (2026-08-26 PDF 본문 재확인).
- `models/{walk,stand}.onnx` 그래프 실측: 입력 `obs [1,40]`, 출력 `mu`/`log_std`/`value`,
  초기화자 `model._model.a2c_network.actor_mlp.{0,2}` = **[1024, 512] ELU**,
  상태무관 `sigma [6]`, 선두에 `Sub`/`Div`/`Clip`(RunningMeanStd 융합) — rl_games 산출물.
- 코드: `src/rl_walk.py`(관측 구성·좌표 변환), `src/rl_stand.py`(FSM 임계값),
  `src/utils.py`(`compute_up_proj`), `src/config.py`(키프레임), `mjcf/chair.xml`.
- Isaac Lab 0.54.4 소스: `direct_rl_env.py:151`(events),
  `isaaclab_rl/rsl_rl/exporter.py:25`(ONNX), `rsl_rl/train.py:99,173`(등록 시점),
  `actuator_pd_cfg.py:52`(`DelayedPDActuatorCfg`), `manager_term_cfg.py:184`(history_length).
- 실측: 2026-08-25 `chair_sim.py --motion script` 헤드리스 로그.
