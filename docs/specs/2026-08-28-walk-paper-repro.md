# walk 1단계 논문 기준 재현 — 설계 (이슈 #18)

> 2026-08-28 브레인스토밍 결과. 상위 스펙 `docs/specs/2026-08-25-isaac-rl-design.md`(이하 "스펙")의
> §3(액션 계약)·§7(학습·합격선)·§8(스택 결정)을 이 문서가 부분 개정한다. 근거는
> `docs/reports/2026-08-28-walk-freeze-analysis.md`(이하 "리포트").

## 목표

리포트 §8 의 손잡이(크기 페널티 발명, 노이즈, entropy 하한)로 가기 **전에**, 논문이 실제로 쓴 학습
스택과 제약을 그대로 재현한다. 제약 안에서 먼저 재현해야 제약을 풀었을 때 무엇이 바뀌었는지 말할
수 있다. 두 팔을 같은 env·같은 자로 돌린다:

- **rl_games 팔** — 논문과 같은 라이브러리. `bounds_loss`·`clip_actions`·관측 정규화 ±5 클립·ONNX
  그래프가 전부 기본값으로 따라온다.
- **rsl_rl + bounds 팔** — #15 의 rsl_rl 경로에 rl_games 의 두 동작(bounds loss, [−1,1] 클램프)만
  이식. 스택 차이 자체를 실험 변수로 본다.

## 종료 조건

두 팔이 논문식 설정으로 §7 규모(≈200 M env-step)를 완주하고, `isaac/scripts/eval.py` 로 잰 합격선
3지표(완주율·전진 속도·좌면 높이) + 포화율 + σ 경과 + 항별 보상이 `docs/reports/` 비교 리포트로
남는다. **걷든 굳든 종료.** 합격선 도달은 실험 결과이지 종료 조건이 아니다.

## 범위 밖

§6 2단계 노이즈(별도 이슈, 이 이슈의 체크포인트에서 이어간다) · §2 system ID · 관절 대응 실측 ·
보상 가중치 튜닝 · entropy/σ 하한 등 "제약 해제" 실험 · stand · `play.py` 플롯.

## 왜 이 방향인가 (사실 확인, 2026-08-28)

| 동작 | rl_games (IsaacGymEnvs Ant/Humanoid PPO yaml) | rsl_rl 5.0.1 |
|---|---|---|
| bounds loss | `bounds_loss_coef: 0.0001` — `soft_bound=1.1`, `clamp_min(mu−1.1,0)² + clamp_max(mu+1.1,0)²` 를 총손실에 가산 (`a2c_continuous.py`) | 없음 |
| 액션 클램프 | `clip_actions=True` 기본 — `clamp(mu,−1,1)` 후 액션 공간으로 rescale, env 는 클램프된 값을 본다 (`a2c_common.py`) | `RslRlVecEnvWrapper(clip_actions=None)` 이면 raw |
| 관측 정규화 | RunningMeanStd, ±5 클립 (`walk.onnx` 그래프에 그대로) | EmpiricalNormalization, 클립 없음 |
| 그 외 (Ant) | normalize_value, value_bootstrap, critic_coef 2, kl 0.008, mini_epochs 4, minibatch 32768, horizon 16 (Humanoid 32) | 대응 필드 있음 |

리포트 §4: 우리 rsl_rl 학습에서 mu 가 2900° 까지 도망가 5/6 관절의 탐색이 죽었다. 위 표의 첫 두 줄이
그 구멍을 막는다. 스펙 §8 이 rsl_rl 을 고른 것이 논문에서 벗어난 지점이었다. 실기 이력의
`ACT_INIT = 1.0` 은 rl_games 정규화 공간 [−1, 1] 의 상한값이다 — 논문 학습이 이 규약이었다는 정황.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| 1 | 두 팔 모두 돌린다 | 스택 차이를 변수로 분리. 비용 2배(각 ~25 분)는 감당 가능 |
| 2 | 액션 규약 = 래퍼가 `[−1, 1]` 클램프, env 는 그 값을 이력에 넣고 관절은 `±0.873` 으로 한 번 더 클립 | 논문(rl_games) 동작. env 무변경. 실기 raw 와의 차이는 bounds loss 가 `|mu| ≤ 1.1` 로 묶어 최대 0.1 rad |
| 3 | 종료 = 두 팔 완주 + 같은 자 보고 | 재현 여부를 답하되 튜닝으로 번지지 않게 |
| 4 | rsl_rl bounds loss 는 `optimizer.step` 훅 (~30줄) | `PPO.update()` 가 200줄 단일 메서드라 복사 없이 넣을 자리가 그것뿐. grad clip 뒤에 gradient 가 더해지는 차이는 1e-4 계수에서 무시 |
| 5 | `eval.py` 를 선행 부품(A0)으로 정식화 | 두 팔 + `walk.onnx` 에 같은 자. 리포트 §9 의 함정을 코드로 고정 |

## 1. 액션 규약 (스펙 §3 개정)

```
policy mu ──clamp(−1, 1)──▶ a ──▶ 이력(obs)                 ← 래퍼 (rl_games: env.clip_actions 1.0 / rsl_rl: clip_actions=1.0)
                              └──clamp(±0.872665)──▶ 관절 목표  ← env (기존, 무변경)
```

- `walk_env.py` 무변경. "받은 액션 → 이력, `ACTION_LIMIT` 클립 → 관절" 그대로.
- `agents/rsl_rl_ppo_cfg.py`: `clip_actions = 1.0`. #15 의 `test_actions_are_not_clipped_by_the_wrapper` 는
  `test_actions_clamped_to_unit_box` 로 뒤집고 근거 주석을 바꾼다.
- 계약 문구 개정: 스펙 §3 "행동", `obs_layout.py` docstring 2번 항목 — "이력에 들어가는 액션은 raw mu"
  → "래퍼가 `[−1,1]` 로 클램프한 mu. 실기(`rl_walk.py`)는 raw 를 넣지만 bounds loss 가 `|mu| ≤ 1.1`
  로 묶으므로 차이 ≤ 0.1 rad. 학습 초기(σ=1)의 이력 분포를 논문과 맞추는 것이 이 클램프의 목적."
- `ACT_INIT = 1.0` 은 그대로 — 이제 규약의 상한값이라는 의미가 생긴다.

## 2. rl_games 팔

**설치.** `pip install -e ~/IsaacLab/source/isaaclab_rl[rl-games]` — isaac-sim 포크
`rl_games.git@python3.11`. `gym 0.23.1` 은 env 에 이미 있다. 설치 전 `pip freeze > logs/pip-before-rl_games.txt`.

**`isaac/chair_rl/agents/rl_games_ppo_cfg.yaml`** — Isaac Lab `direct/ant/agents/rl_games_ppo_cfg.yaml` 을
뼈대로. 값의 출처 표기는 #15 와 같은 세 분류.

| 키 | 값 | 출처 |
|---|---|---|
| `env.clip_actions` | 1.0 | 결정 2 |
| `env.clip_observations` | 5.0 | [측정] `walk.onnx` 의 Clip(−5, 5) |
| `network.mlp.units` / activation | `[1024, 512]` / elu | [측정] `walk.onnx` |
| `fixed_sigma` / `sigma_init` | True / const 0 | [측정] 상태무관 `sigma[6]`, exp(0)=1 |
| `normalize_input` / `normalize_value` / `value_bootstrap` | True / True / True | [기본] Ant |
| `gamma` / `tau` / `e_clip` / `entropy_coef` | 0.99 / 0.95 / 0.2 / 0.0 | [기본] Ant (= 스펙 §7) |
| `learning_rate` / `lr_schedule` / `kl_threshold` | 3e-4 / adaptive / 0.008 | [기본] Ant |
| `critic_coef` / `mini_epochs` / `minibatch_size` | 2 / 4 / 32768 | [기본] Ant |
| **`bounds_loss_coef`** | **1e-4** | [기본] Ant·Humanoid 공통 — 이 이슈의 핵심 |
| `horizon_length` | 32 | [가정] 스펙 §7 결정 유지 (Ant 16 / Humanoid 32) |
| `max_epochs` | 1500 | [가정] 4096 × 32 × 1500 = 196.6 M env-step, #15 와 동일 |
| `reward_shaper.scale_value` | 1.0 | [가정] 논문 미상. Isaac Lab Ant 0.6, IGE Humanoid 0.01 — Table IV 값을 그대로 보려고 1.0 |
| `grad_norm` / `truncate_grads` | 1.0 / True | [기본] |
| `save_frequency` / `save_best_after` | 50 / 100 | 운영 |

**등록.** `chair_rl/__init__.py` kwargs 에 `"rl_games_cfg_entry_point": "chair_rl.agents:rl_games_ppo_cfg.yaml"`.
문자열 + yaml 이라 Kit 밖 import 없음. `pyproject.toml` 의 `package-data` 에 yaml 포함.

**`isaac/scripts/rl_games/train.py`.** #15 shim 과 같은 방식. upstream 경로 탐색(`ISAACLAB_PATH` →
`isaaclab` 모듈 위치)은 `isaac/scripts/_upstream.py` 로 빼고 두 shim 이 공유한다. hydra exit-0 함정
주석도 공유.

## 3. rsl_rl + bounds 팔

**`isaac/chair_rl/agents/bounded_ppo.py`**

```python
class BoundedPPO(PPO):
    """rl_games 의 bound_loss 를 rsl_rl PPO 에 이식. soft_bound 1.1, 계수 bounds_loss_coef.
    update() 는 200줄 단일 메서드라 끊어 넣을 자리가 없어 optimizer.step 을 감싼다 —
    actor forward 훅이 마지막 minibatch 의 output_mean 을 잡아 두고, step 직전에
    bound_loss.backward() 로 gradient 를 누적한다. rl_games 와의 차이: grad clip 이 bound
    gradient 보다 먼저 걸린다 (1e-4 계수에서 무시)."""
    def __init__(self, *args, bounds_loss_coef: float = 1e-4, **kwargs): ...
```

- cfg: `algorithm = RslRlPpoAlgorithmCfg(class_name="chair_rl.agents.bounded_ppo:BoundedPPO", ...)` —
  rsl_rl `resolve_callable` 이 `module:Class` 를 받는다 (`utils.py:97`). `bounds_loss_coef` 는
  `RslRlPpoAlgorithmCfg` 에 없는 필드라 configclass 서브클래스 `BoundedPpoAlgorithmCfg` 로 추가.
- 로그: `bound_loss` 평균을 `extras`/텐서보드에 남긴다 — 포화 여부를 학습 중에 본다.
- 나머지 하이퍼파라미터는 §2 표를 rsl_rl 필드로 매핑: `num_steps_per_env 32`, `num_mini_batches 4`
  (= 4096×32/32768), `num_learning_epochs 4`, `desired_kl 0.008`, `value_loss_coef 1.0`
  (rl_games `0.5 × critic_coef 2`), `clip_actions 1.0`. #15 의 5/0.01/None 에서 바뀌는 셋은 커밋 메시지에 명시.

## 4. `isaac/scripts/eval.py` (스펙 §7 "평가 하네스" 구현)

```
eval.py --task Chair-Walk-Direct-v0 --headless --num_envs 64 --seed 123 \
        (--rsl_rl CKPT.pt | --rl_games CKPT.pth | --onnx models/walk.onnx [--joint_order tree|actuator|tree-reversed] [--quat_flip]) \
        --steps 340 --out result.json
```

출력 JSON: `{완주율, 전진속도_mps, 좌면높이_m, x변위_m, y변위_m, 항별_보상_per_step{progress,height,up,heading,action,vel,total}, 포화율{평균, 관절별}, sigma[6] (있으면), 평균_관절각_deg[6], 평균_pitch_deg, steps, seed, ckpt}`.

측정 정의 = 리포트 §9 그대로. 코드로 고정하는 함정:
- `--steps` 기본 340 (< `MAX_EPISODE_LEN` 350) — truncation 리셋이 x·y 를 0 으로 되돌리는 허수 방지.
  350 이상을 주면 경고.
- `reset_terminated` 인 env 는 그 스텝부터 집계 제외(`alive` 마스크는 한 번 꺼지면 안 켜진다).
- 롤아웃 전체(`reset()` 포함)를 `torch.inference_mode()` 안에서.
- obs 는 TensorDict — `obs["policy"]`.
- ONNX 는 `onnx.numpy_helper` 로 초기화자를 읽어 torch 로 재구현하고 onnxruntime 과 대조(오차 <1e-5 assert).
- 보상 항은 `mdp.walk_reward_terms` 를 직접 호출해 `cfg.reward_weights` 로 가중.

## 5. 학습량과 실행

두 팔 모두 4096 env × horizon 32 × 1500 = 196.6 M env-step (스펙 §7 "walk 200 M ± 2배", #15 와 동일).

```bash
PYTHONPATH= python isaac/scripts/rl_games/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 4096 --max_iterations 1500
PYTHONPATH= python isaac/scripts/rsl_rl/train.py   --task Chair-Walk-Direct-v0 --headless --num_envs 4096 --max_iterations 1500
PYTHONPATH= python isaac/scripts/eval.py --rl_games logs/rl_games/chair_walk/<run>/nn/last.pth --out ...
PYTHONPATH= python isaac/scripts/eval.py --rsl_rl   logs/rsl_rl/chair_walk/<run>/model_1499.pt --out ...
PYTHONPATH= python isaac/scripts/eval.py --onnx models/walk.onnx --joint_order tree --out ...
```

`--video` 는 #15 측정대로 −12 % 라 켜도 된다. 판정은 로그의 `Learning iteration`(hydra exit 0).

## 6. 테스트

| 테스트 | Kit | 검증 |
|---|---|---|
| `test_registry.py` 확장 | 불필요 | `rl_games_cfg_entry_point` 가 문자열이고 `chair_rl.agents.*` 가 import 되지 않았다 |
| `test_bounded_ppo.py` | 불필요 (torch) | 더미 actor 에 훅을 걸고 mu=±0.5 → bound gradient 0, mu=±2 → 0 아님, 계수 비례 |
| `test_agent_cfg.py` 수정 | 필요 | `clip_actions == 1.0`, `algorithm.class_name` 이 BoundedPPO, `bounds_loss_coef == 1e-4` |
| rl_games 스모크 | 필요 | `--num_envs 64 --max_iterations 2` 가 `Learning iteration` 두 번 + 체크포인트 |
| `test_eval_harness.py` | 필요 | `a_stand` 고정 정책으로 eval 하면 완주 64/64, 전진 ≈ 0, 높이 ≈ 0.080, total ≈ 27.96 (리포트 §3 값) |

## 7. 이슈·PR 분할 (400줄 룰)

| | 내용 | 규모 | 브랜치 |
|---|---|---|---|
| A0 | `eval.py` + `test_eval_harness.py` | ~150 | `feat/18-eval-harness` |
| A1 | 액션 규약: rsl_rl cfg `clip_actions=1.0`, 테스트 뒤집기, 스펙 §3·docstring 문구 | ~40 | `feat/18-action-clamp` |
| A2 | rl_games 팔: 설치, yaml, 등록, `_upstream.py` + shim, 스모크 | ~130 | `feat/18-rl-games` |
| A3 | rsl_rl 팔: `bounded_ppo.py`, cfg 서브클래스, 단위 테스트 | ~60 | `feat/18-bounded-ppo` |
| A4 | 두 팔 전량 학습 + `docs/reports/2026-08-xx-walk-paper-repro.md` | 문서 | `docs/18-walk-paper-repro-report` |

전부 `feat/15-train-path` 위에 쌓는다(#15 미머지, PR base 도 거기). 서브 이슈는 만들지 않는다 —
브랜치 다섯 개가 모두 #18 하나에 대응한다(CLAUDE.md 규칙은 "브랜치는 이슈 하나에만", 역은 아니다).
A0→A1→(A2 ∥ A3)→A4 순서.

## 8. 위험과 열린 것

- **rl_games 설치가 env 를 건드린다.** git 브랜치 의존, `gym` 구버전. 실패하면 A2 만 멈추고 A3 로 간다.
- **`reward_shaper 1.0` 은 가정.** 논문이 IGE 기본(Ant 0.6?)을 썼을 수 있다. 바꾸면 Table IV 의 절대값
  해석이 흔들리므로 1.0 으로 시작하고 A4 리포트에 남긴다.
- **A3 의 grad-clip 순서 차이.** 1e-4 에서 무시한다고 봤지만 측정하지 않았다. `bound_loss` 로그로 두 팔의 값을 비교한다.
- **논문이 걸은 이유 3후보(리포트 §6)는 이 이슈로 하나만 닫힌다** — 스택. dt 와 131k env 는 그대로 열려 있다.
- 두 팔 다 굳으면: 리포트 §8-2(정지가 안정하지 않게 — §6 노이즈)가 다음 이슈다. 한 팔만 걸으면 스택 차이가
  원인이라는 강한 신호.

## 9. 이 설계가 틀렸다면 어떻게 틀렸을지

- **클램프가 폭주를 막는다고 착각했을 수 있다.** 클램프는 관측·비용이 보는 값을 묶을 뿐 mu 자체를 묶지 않는다.
  mu 를 묶는 건 bounds loss 하나다. 두 동작을 함께 넣는 이유가 그것이고, bounds loss 없이 클램프만 있으면
  #15 와 같은 결과가 나와야 한다 — A3 의 단위 테스트가 아니라 A4 의 포화율 곡선이 이 문장을 검증한다.
- **rl_games 의 액션 공간 rescale 을 놓쳤을 수 있다.** `rescale_actions(low, high, clamp(mu))` 는 env 액션
  공간이 [−1,1] 이 아니면 스케일이 바뀐다. Isaac Lab 래퍼는 `Box(−clip_actions, clip_actions)` 를 액션
  공간으로 보고하므로 1.0 이면 항등이다 — A2 스모크에서 `env.action_space` 를 찍어 확인한다.
- **논문 이력이 클램프값이었다는 것은 정황(`ACT_INIT=1.0`)이지 증거가 아니다.** 틀렸다면 실기 이식 시
  이력 분포가 어긋나는 쪽은 우리다 — 차이는 bounds loss 덕에 ≤0.1 rad 라 실기 검증 단계에서 잡힌다.
- **horizon 32 는 여전히 가정이다.** Ant 는 16. 두 팔을 같은 값으로 두므로 팔 간 비교에는 영향이 없지만
  "논문 재현" 으로서는 오차다.
