# walk 1단계 논문 기준 재현 구현 계획 (이슈 #18)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rl_games 팔과 rsl_rl+bounds 팔이 논문식 설정(정책 출력 [−1,1] 클램프, bounds loss, Ant PPO 하이퍼파라미터, [1024,512] ELU, 관측 정규화)으로 각각 ≈200 M env-step 을 완주하고, `scripts/eval.py` 로 잰 같은 자의 결과가 `docs/reports/` 에 남는다.

**Architecture:** env(`walk_env.py`)는 손대지 않는다 — 클램프는 래퍼, bounds loss 는 알고리즘, 측정은 `chair_rl/evaluate.py`(Kit 뒤에 import 되는 순수 함수) + `chair_rl/mlp_policy.py`(rl_games `.pth`·ONNX 를 같은 torch 모듈로) + `scripts/eval.py`(CLI). rl_games 는 Isaac Lab 의 래퍼·train.py 를 shim 으로 그대로 쓴다(#15 방식, 공통부는 `scripts/_upstream.py`). rsl_rl 팔은 `BoundedPPO(PPO)` 가 actor 의 `output_mean` 텐서에 gradient 훅을 걸어 rl_games `bound_loss` 의 미분을 같은 backward 안에 더한다 — grad clip 순서까지 rl_games 와 같다.

**Tech Stack:** Isaac Lab 0.54.4, rsl-rl-lib 5.0.1, rl_games(isaac-sim 포크 `python3.11` 브랜치, A2 에서 설치), torch, onnx/onnxruntime, pytest(`--isaac` 로 Kit 테스트).

**Spec:** `docs/specs/2026-08-28-walk-paper-repro.md` (이하 "스펙"). 근거 리포트 `docs/reports/2026-08-28-walk-freeze-analysis.md`. 상위 스펙 `docs/specs/2026-08-25-isaac-rl-design.md` §3·§7·§8.

## Global Constraints

- 브랜치 하나 = 이슈 하나. 이 계획의 브랜치는 전부 #18 에 대응: `feat/18-eval-harness` → `feat/18-action-clamp` → `feat/18-bounded-ppo` → `feat/18-rl-games` → `docs/18-walk-paper-repro-report`. **베이스는 전부 `feat/15-train-path`** (#15 미머지, PR base 도 거기). 앞 태스크 브랜치가 머지되지 않았으면 그 브랜치 위에 쌓는다. (CLAUDE.md, 스펙 §7)
- 순서 **A0 → A1 → A3 → A2 → A4**. A2(rl_games 설치)가 env 를 건드리므로 A3 뒤에. (스펙 §7)
- diff 400줄 초과 시 멈추고 분할안. 계획 밖 문제는 고치지 말고 보고 — 라벨 깊이초과/선행조건/전제무효화/인접유혹. PR 본문에 "이 코드가 틀렸다면 어떻게 틀렸을지" 필수. (CLAUDE.md)
- `walk_env.py` **무변경**. 클램프는 래퍼(`clip_actions=1.0`), 관절 클립 `±0.872665` 는 env 가 이미 한다. (스펙 §1)
- 값 출처 표기 `[측정]`/`[기본]`/`[가정]` (#15 관례). `bounds_loss_coef 1e-4`, `soft_bound 1.1`, `horizon 32`, `minibatch 32768`, `mini_epochs 4`, `kl 0.008`, `critic_coef 2`, `reward_shaper 1.0`, `clip_observations 5.0`. (스펙 §2)
- 실행 환경: conda `env_isaaclab`(`~/miniforge3/envs/env_isaaclab/bin/python`), 저장소 루트, **`PYTHONPATH=` 비움**. CPU 테스트 `pytest isaac -q`, Kit 테스트 `pytest isaac/tests/<file> --isaac -q`. Kit 을 띄우는 스크립트는 `python -u` — 종료 시 stdout 버퍼가 날아간다.
- 학습 실행 판정은 종료코드가 아니라 로그의 `Learning iteration`(rsl_rl) / `epoch:`(rl_games) — hydra 가 예외를 삼키고 exit 0 을 낸다. (스펙 §5)
- 평가 롤아웃은 **340 스텝**(< `MAX_EPISODE_LEN` 350), `reset()` 포함 전체를 `torch.inference_mode()` 안에서, obs 는 TensorDict(`obs["policy"]`). (스펙 §4)
- env 의 private 속성(`_a2j`, `_p_target`, `_root_quat_xyzw`)을 하네스에서 쓰지 않는다 — #12 리뷰 라운드 1 결정. 공개 API 로 다시 만든다.
- 커밋 `feat[isaac]: …` / `test[isaac]: …` / `docs[isaac]: …`, 트레일러 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Kit 테스트는 모듈당 env 하나, 픽스처 정리는 `wrapped.close()` **뒤에** `sim.stop()` (`test_env_smoke.py` docstring).

---

## 파일 구조

| 파일 | 책임 | 태스크 |
|---|---|---|
| `isaac/chair_rl/mlp_policy.py` (신규) | rl_games `.pth` / ONNX → 같은 torch MLP(정규화+ELU×2+mu). torch·onnx 만 의존 | 1 |
| `isaac/chair_rl/evaluate.py` (신규) | 롤아웃·지표 계산 순수 함수, 정책 어댑터(상수/rsl_rl/MLP/관절순서 치환). Kit 뒤 import | 1 |
| `isaac/scripts/eval.py` (신규) | CLI: AppLauncher → env → 정책 선택 → `evaluate.rollout` → JSON | 1 |
| `isaac/tests/test_mlp_policy.py` (신규, CPU) | 합성 state_dict / `models/walk.onnx` vs onnxruntime | 1 |
| `isaac/tests/test_eval_harness.py` (신규, Kit) | `a_stand` 고정 정책 → 리포트 §3 값 | 1 |
| `isaac/chair_rl/agents/rsl_rl_ppo_cfg.py` | `clip_actions=1.0`, `BoundedPpoAlgorithmCfg`, Ant 매핑 | 2, 3 |
| `isaac/tests/test_agent_cfg.py` | 계약 테스트 뒤집기·확장 | 2, 3 |
| `isaac/chair_rl/obs_layout.py` | docstring 2번 계약 문구 | 2 |
| `docs/specs/2026-08-25-isaac-rl-design.md` §3 행동 | 계약 개정 블록 | 2 |
| `isaac/chair_rl/agents/bounded_ppo.py` (신규) | `bound_loss`, `attach_bound_loss`, `BoundedPPO` | 3 |
| `isaac/tests/test_bounded_ppo.py` (신규, CPU) | 해석적 gradient 검증 | 3 |
| `isaac/scripts/_upstream.py` (신규) | upstream 경로 탐색 + `run_upstream(subdir)` | 4 |
| `isaac/scripts/rsl_rl/train.py`, `isaac/scripts/rl_games/train.py` (신규) | 3줄 shim | 4 |
| `isaac/chair_rl/agents/rl_games_ppo_cfg.yaml` (신규) | rl_games PPO cfg | 4 |
| `isaac/chair_rl/__init__.py`, `isaac/pyproject.toml`, `isaac/tests/test_registry.py` | 엔트리포인트·package-data·등록 테스트 | 4 |
| `docs/reports/2026-08-XX-walk-paper-repro.md` (신규) | 두 팔 비교 리포트 | 5 |

---

### Task 1 (A0): 평가 하네스 — `mlp_policy.py`, `evaluate.py`, `scripts/eval.py`

브랜치 `feat/18-eval-harness` (`git checkout -b feat/18-eval-harness feat/15-train-path`).

**Files:**
- Create: `isaac/chair_rl/mlp_policy.py`, `isaac/chair_rl/evaluate.py`, `isaac/scripts/eval.py`
- Test: `isaac/tests/test_mlp_policy.py`, `isaac/tests/test_eval_harness.py`

**Interfaces:**
- Produces `mlp_policy.MlpPolicy(nn.Module)`: `forward(obs (N,40)) -> mu (N,6)`; `MlpPolicy.from_rl_games(path, device="cpu")`; `MlpPolicy.from_onnx(path, device="cpu")`; 속성 `sigma: Tensor(6) | None`.
- Produces `evaluate.ActFn = Callable[[TensorDict], Tensor]` (입력 = 래퍼 obs, 출력 = mu (N,6) env 정책 순서), `evaluate.constant_policy(values) -> ActFn`, `evaluate.tensor_policy(model) -> ActFn`, `evaluate.permuted(act, order, quat_flip) -> ActFn`, `evaluate.rollout(raw, env, act, steps=340, clip_actions=1.0) -> dict`, `evaluate.JOINT_ORDERS`.
- Task 4 의 스모크와 Task 5 가 `scripts/eval.py --rsl_rl|--rl_games|--onnx|--a_stand` 를 쓴다.

- [ ] **Step 1: `mlp_policy.py` CPU 테스트를 쓴다**

`isaac/tests/test_mlp_policy.py`:

```python
"""rl_games 체크포인트와 ONNX 를 같은 torch 모듈로 읽는지. Kit 불필요."""

import math

import pytest
import torch

from chair_rl.mlp_policy import MlpPolicy

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

WALK_ONNX = "models/walk.onnx"


def _synthetic_rl_games_ckpt(tmp_path, obs_dim=40, act_dim=6, units=(8, 4)):
    """rl_games ModelA2CContinuousLogStd 의 state_dict 키를 그대로 흉내 낸다."""
    g = torch.Generator().manual_seed(0)
    sd = {
        "running_mean_std.running_mean": torch.randn(obs_dim, generator=g, dtype=torch.float64),
        "running_mean_std.running_var": torch.rand(obs_dim, generator=g, dtype=torch.float64) + 0.5,
        "running_mean_std.count": torch.tensor(1000.0, dtype=torch.float64),
        "a2c_network.sigma": torch.full((act_dim,), math.log(0.3)),
    }
    dims = [obs_dim, *units]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        sd[f"a2c_network.actor_mlp.{2 * i}.weight"] = torch.randn(b, a, generator=g) * 0.1
        sd[f"a2c_network.actor_mlp.{2 * i}.bias"] = torch.randn(b, generator=g) * 0.1
    sd["a2c_network.mu.weight"] = torch.randn(act_dim, units[-1], generator=g) * 0.1
    sd["a2c_network.mu.bias"] = torch.zeros(act_dim)
    sd["a2c_network.value.weight"] = torch.randn(1, units[-1], generator=g)
    sd["a2c_network.value.bias"] = torch.zeros(1)
    path = tmp_path / "last_chair_walk_ep_2_rew_1.pth"
    torch.save({"model": sd, "epoch": 2}, path)
    return path, sd


def test_from_rl_games_matches_manual_forward(tmp_path):
    path, sd = _synthetic_rl_games_ckpt(tmp_path)
    pol = MlpPolicy.from_rl_games(path)
    x = torch.randn(5, 40)
    # rl_games RunningMeanStd: (x - mean) / sqrt(var + 1e-5), clamp ±5
    h = (x - sd["running_mean_std.running_mean"].float()) / torch.sqrt(sd["running_mean_std.running_var"].float() + 1e-5)
    h = h.clamp(-5.0, 5.0)
    h = torch.nn.functional.elu(h @ sd["a2c_network.actor_mlp.0.weight"].T + sd["a2c_network.actor_mlp.0.bias"])
    h = torch.nn.functional.elu(h @ sd["a2c_network.actor_mlp.2.weight"].T + sd["a2c_network.actor_mlp.2.bias"])
    want = h @ sd["a2c_network.mu.weight"].T + sd["a2c_network.mu.bias"]
    torch.testing.assert_close(pol(x), want)
    torch.testing.assert_close(pol.sigma, torch.full((6,), 0.3))


def test_from_onnx_matches_onnxruntime():
    pol = MlpPolicy.from_onnx(WALK_ONNX)
    sess = ort.InferenceSession(WALK_ONNX, providers=["CPUExecutionProvider"])
    x = torch.randn(7, 40, generator=torch.Generator().manual_seed(1))
    for row in x:                                   # 그래프 입력이 [1, 40] 고정
        ref = sess.run(["mu"], {"obs": row[None].numpy()})[0]
        torch.testing.assert_close(pol(row[None]), torch.tensor(ref), atol=1e-5, rtol=1e-4)
    # 논문 정책의 최종 σ (리포트 §6): 0.046–0.091
    assert pol.sigma is not None and 0.04 < pol.sigma.min() and pol.sigma.max() < 0.1
```

- [ ] **Step 2: 실패 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_mlp_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chair_rl.mlp_policy'`

- [ ] **Step 3: `mlp_policy.py` 구현**

`isaac/chair_rl/mlp_policy.py`:

```python
"""rl_games 형 MLP 정책 하나로 두 파일 형식을 읽는다 (스펙 #18 §4).

구조는 models/walk.onnx 그래프 그대로:
    (obs − mean) / std → clip(±5) → [Linear → ELU] × k → Linear → mu
- rl_games .pth: torch.load(...)["model"] 의 키 `running_mean_std.{running_mean,running_var}`,
  `a2c_network.actor_mlp.{0,2,...}.{weight,bias}`, `a2c_network.mu.*`, `a2c_network.sigma`.
  정규화는 RunningMeanStd.forward — (x − mean)/sqrt(var + 1e-5) 뒤 clamp(−5, 5).
- ONNX: Sub/Div 의 두 번째 입력이 mean/std(std 는 sqrt(var+eps) 가 미리 계산된 값), Clip 의
  상수가 경계, Elu 로 이어지는 Gemm 이 은닉층, 출력 이름 "mu" 인 Gemm 이 헤드.
활성함수는 ELU 로 고정한다 — [측정] walk.onnx. 다른 활성함수의 체크포인트는 여기서 읽지 않는다.
torch 와 onnx 만 의존한다. Kit 없음.
"""

from __future__ import annotations

import pathlib

import torch
from torch import nn

RL_GAMES_EPS = 1e-5
RL_GAMES_CLIP = 5.0


class MlpPolicy(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hidden: list[tuple[torch.Tensor, torch.Tensor]],
                 mu: tuple[torch.Tensor, torch.Tensor], clip: tuple[float, float] = (-RL_GAMES_CLIP, RL_GAMES_CLIP),
                 log_sigma: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.clip = clip
        self.hidden = nn.ModuleList()
        for w, b in hidden:
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w.float(), b.float()
            self.hidden.append(lin)
        w, b = mu
        self.mu = nn.Linear(w.shape[1], w.shape[0])
        self.mu.weight.data, self.mu.bias.data = w.float(), b.float()
        self.register_buffer("log_sigma", None if log_sigma is None else log_sigma.float())

    @property
    def sigma(self) -> torch.Tensor | None:
        return None if self.log_sigma is None else torch.exp(self.log_sigma)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = ((obs - self.mean) / self.std).clamp(*self.clip)
        for lin in self.hidden:
            x = nn.functional.elu(lin(x))
        return self.mu(x)

    @classmethod
    def from_rl_games(cls, path: str | pathlib.Path, device: str = "cpu") -> "MlpPolicy":
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        mean = sd["running_mean_std.running_mean"]
        std = torch.sqrt(sd["running_mean_std.running_var"].float() + RL_GAMES_EPS)
        hidden, i = [], 0
        while f"a2c_network.actor_mlp.{i}.weight" in sd:
            hidden.append((sd[f"a2c_network.actor_mlp.{i}.weight"], sd[f"a2c_network.actor_mlp.{i}.bias"]))
            i += 2                                   # Sequential: Linear, ELU, Linear, ELU, ...
        mu = (sd["a2c_network.mu.weight"], sd["a2c_network.mu.bias"])
        return cls(mean, std, hidden, mu, log_sigma=sd.get("a2c_network.sigma")).to(device)

    @classmethod
    def from_onnx(cls, path: str | pathlib.Path, device: str = "cpu") -> "MlpPolicy":
        import onnx
        from onnx import numpy_helper

        g = onnx.load(str(path)).graph
        init = {t.name: torch.tensor(numpy_helper.to_array(t)) for t in g.initializer}
        const = {n.output[0]: float(numpy_helper.to_array(n.attribute[0].t))
                 for n in g.node if n.op_type == "Constant"}
        node = {op: [n for n in g.node if n.op_type == op] for op in ("Sub", "Div", "Clip", "Gemm", "Elu")}
        mean, std = init[node["Sub"][0].input[1]], init[node["Div"][0].input[1]]
        clip_node = node["Clip"][0]
        clip = (const[clip_node.input[1]], const[clip_node.input[2]])
        elu_inputs = {n.input[0] for n in node["Elu"]}
        hidden = [(init[n.input[1]], init[n.input[2]]) for n in node["Gemm"] if n.output[0] in elu_inputs]
        mu_node = next(n for n in node["Gemm"] if n.output[0] == "mu")
        mu = (init[mu_node.input[1]], init[mu_node.input[2]])
        sigma = next((v for k, v in init.items() if k.endswith(".sigma")), None)
        return cls(mean, std, hidden, mu, clip=clip, log_sigma=sigma).to(device)
```

- [ ] **Step 4: 통과 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_mlp_policy.py -q`
Expected: `2 passed`

- [ ] **Step 5: 커밋**

```bash
git add isaac/chair_rl/mlp_policy.py isaac/tests/test_mlp_policy.py
git commit -m "feat[isaac]: mlp_policy — rl_games .pth 와 walk.onnx 를 같은 torch MLP 로 (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: `evaluate.py` 구현** (Kit 테스트는 Step 7 — 순수 함수 부분부터 쓴다)

`isaac/chair_rl/evaluate.py`:

```python
"""평가 하네스의 본체 (스펙 #18 §4). scripts/eval.py 는 이걸 부르는 CLI 다.

Kit 이 뜬 뒤에만 import 한다 — isaaclab 은 함수 안에서 늦게 import 한다. 리포트 §9 의 함정을
코드로 고정한다:
  1. steps < MAX_EPISODE_LEN — truncation 리셋이 x·y 를 정확히 0 으로 되돌려 "변위 0" 허수를 만든다.
  2. terminated 로 리셋된 env 는 그 스텝부터 집계 제외. alive 는 한 번 꺼지면 켜지지 않는다.
  3. reset() 포함 전체가 torch.inference_mode() 안 — 밖이면 write_joint_state_to_sim 이
     inference tensor 갱신으로 죽는다.
  4. 래퍼 obs 는 TensorDict — obs["policy"].
정책 어댑터는 전부 ActFn = (TensorDict) -> mu (N, 6), env 정책 순서(ol.POLICY_JOINT_NAMES).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from . import mdp
from . import obs_layout as ol

ActFn = Callable[[object], torch.Tensor]

# chair_sim.JOINT_ORDERS 와 같은 세 후보. chair_sim 은 모듈 최상위에서 Isaac 을 import 하고
# argparse 를 돌리므로 여기서 가져오지 않는다 (인접유혹 — chair_sim 정리는 이 이슈 밖).
JOINT_ORDERS = {
    "tree": ("joint2", "joint1", "joint4", "joint3", "joint6", "joint5"),
    "actuator": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    "tree-reversed": ("joint5", "joint6", "joint3", "joint4", "joint1", "joint2"),
}


def constant_policy(values: Sequence[float]) -> ActFn:
    v = torch.tensor(values, dtype=torch.float32)

    def act(obs):
        o = obs["policy"]
        return v.to(o.device).expand(o.shape[0], -1)
    return act


def tensor_policy(model: Callable[[torch.Tensor], torch.Tensor]) -> ActFn:
    """obs["policy"] 텐서를 받는 모델(MlpPolicy 등)을 ActFn 으로."""
    return lambda obs: model(obs["policy"])


def permuted(act_on_tensor: Callable[[torch.Tensor], torch.Tensor], order: Sequence[str],
             quat_flip: bool = False) -> ActFn:
    """다른 관절 순서·쿼터니언 부호로 학습된 정책(논문 walk.onnx)을 env 순서로 감싼다.

    order[i] = 정책 출력 i 가 가는 관절. env 슬롯 j 는 관절 POLICY_JOINT_NAMES[j] = order[perm[j]].
    - 출력: mu_env[:, j] = mu_policy[:, perm[j]]
    - 액션 이력(obs[16:]): 정책 슬롯 i 에는 env 슬롯 inv[i] 의 값 (inv = perm 의 역)
    - quat_flip: 실기 IMU 규약 (−x, −y, z, w) 로 학습됐을 가능성 — 쿼터니언 이력 4개의 x, y 부호 반전
    """
    perm = [tuple(order).index(n) for n in ol.POLICY_JOINT_NAMES]
    inv = [perm.index(i) for i in range(ol.NUM_ACTIONS)]

    def act(obs):
        o = obs["policy"].clone()
        if quat_flip:
            o[:, 0:16:4] *= -1
            o[:, 1:16:4] *= -1
        if perm != list(range(ol.NUM_ACTIONS)):
            ah = o[:, 16:].view(-1, ol.NUM_ACT_HIS, ol.NUM_ACTIONS)[:, :, inv]
            o = torch.cat([o[:, :16], ah.reshape(o.shape[0], -1)], dim=1)
        return act_on_tensor(o)[:, perm]
    return act


def rollout(raw, env, act: ActFn, steps: int = 340, clip_actions: float | None = 1.0) -> dict:
    """raw = DirectRLEnv(unwrapped), env = 그 위의 RslRlVecEnvWrapper(clip_actions 동일값).

    반환 dict 의 키는 scripts/eval.py 의 JSON 과 같다. 보상 항은 env 의 extras 가 아니라
    mdp.walk_reward_terms 를 직접 불러 alive 마스크로 누적한다 — extras 는 리셋된 env 를 포함한 평균이다.
    """
    if steps >= mdp.MAX_EPISODE_LEN:
        raise ValueError(f"steps={steps} 는 MAX_EPISODE_LEN({mdp.MAX_EPISODE_LEN}) 미만이어야 한다 — "
                         "truncation 리셋이 위치를 0 으로 되돌려 변위가 허수가 된다")
    W = raw.cfg.reward_weights
    n, dev = raw.num_envs, raw.device
    a2j, _ = raw.robot.find_joints(list(ol.POLICY_JOINT_NAMES), preserve_order=True)
    p_target = torch.tensor(mdp.P_TARGET, device=dev)
    keys = ("progress", "height", "up", "heading", "action", "vel")
    weights = {"progress": W.progress, "height": W.height, "up": W.up, "heading": W.heading,
               "action": W.action, "vel": W.vel}

    with torch.inference_mode():
        raw.reset()
        obs = env.get_observations()
        p0 = raw.root_pos_local().clone()
        p_last = p0.clone()
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        prev_a = torch.full((n, ol.NUM_ACTIONS), ol.ACT_INIT, device=dev)
        pot = mdp.potential(p0, p_target)
        acc = {k: torch.zeros(n, device=dev) for k in (*keys, "total")}
        s_alive = torch.zeros(n, device=dev)
        z_sum = torch.zeros(n, device=dev)
        pitch_sum = torch.zeros(n, device=dev)
        jpos_sum = torch.zeros(n, ol.NUM_ACTIONS, device=dev)
        sat_unit = torch.zeros(n, device=dev)
        sat_joint = torch.zeros(n, device=dev)
        for _ in range(steps):
            mu = act(obs)
            a = mu.clamp(-clip_actions, clip_actions) if clip_actions is not None else mu   # 래퍼와 같은 값
            obs, _, _, _ = env.step(mu)                                                     # 래퍼가 클램프한다
            p = raw.root_pos_local()
            q = ol.wxyz_to_xyzw(raw.robot.data.root_quat_w)
            terms, pot = mdp.walk_reward_terms(p, q, pot, a, prev_a, raw.robot.data.joint_vel, p_target=p_target)
            prev_a = a
            total = sum(weights[k] * terms[k] for k in keys) + W.alive
            m = alive & ~raw.reset_terminated.bool()
            mf = m.float()
            for k in keys:
                acc[k] += terms[k] * mf
            acc["total"] += total * mf
            s_alive += mf
            p_last = torch.where(m[:, None], p, p_last)
            z_sum += p[:, 2] * mf
            pitch = torch.asin((2.0 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0])).clamp(-1.0, 1.0))
            pitch_sum += pitch * mf
            jpos_sum += raw.robot.data.joint_pos[:, a2j] * mf[:, None]
            sat_unit += (mu.abs() > 1.0).float().mean(-1) * mf
            sat_joint += (mu.abs() > ol.ACTION_LIMIT).float().mean(-1) * mf
            alive = m

    s = s_alive.clamp_min(1.0)
    d = p_last - p0
    per_step = {k: (acc[k] / s).mean().item() for k in acc}
    return {
        "steps": steps,
        "num_envs": n,
        "completion_rate": alive.float().mean().item(),
        "forward_speed_mps": (d[:, 0] / (s * mdp.CONTROL_DT)).mean().item(),
        "x_disp_m": d[:, 0].mean().item(),
        "y_disp_m": d[:, 1].mean().item(),
        "seat_height_m": (z_sum / s).mean().item(),
        "pitch_deg": math.degrees((pitch_sum / s).mean().item()),
        "joint_pos_deg": [math.degrees(v) for v in (jpos_sum / s[:, None]).mean(0).tolist()],
        "reward_per_step": per_step,                                   # 가중치 미적용 항 + total(가중 합)
        "weighted_per_step": {k: weights[k] * per_step[k] for k in keys},
        "saturation": {"unit": (sat_unit / s).mean().item(), "joint": (sat_joint / s).mean().item()},
    }
```

- [ ] **Step 7: Kit 테스트 — `a_stand` 고정 정책이 리포트 §3 값을 내는가**

`isaac/tests/test_eval_harness.py`:

```python
"""evaluate.rollout 이 알려진 정책에 알려진 값을 내는지. Kit 필요:
    pytest isaac/tests/test_eval_harness.py --isaac -v

기준값은 리포트 2026-08-28 §3 (64 env, seed 123): a_stand 고정 → 완주 64/64, 변위 0,
z 0.080, 보상/스텝 27.96 (h 20 + up 5 + hd 2 + alive 1 − act 0.04). env 는 모듈당 하나,
정리 순서는 test_env_smoke.py docstring 대로 close() 뒤 sim.stop().
"""

import pytest

from chair_rl import mdp

pytestmark = pytest.mark.isaac

NUM_ENVS = 16


@pytest.fixture(scope="module")
def envs(kit_app):
    import gymnasium as gym

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import chair_rl  # noqa: F401
    from chair_rl.walk_env import WalkEnvCfg

    cfg = WalkEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 123
    raw = gym.make("Chair-Walk-Direct-v0", cfg=cfg).unwrapped
    sim = raw.sim
    yield raw, RslRlVecEnvWrapper(raw, clip_actions=1.0)
    raw.close()
    sim.stop()


def test_a_stand_hold_matches_report_baseline(envs):
    from chair_rl import evaluate

    raw, env = envs
    r = evaluate.rollout(raw, env, evaluate.constant_policy(mdp.A_STAND), steps=340)
    assert r["completion_rate"] == 1.0
    assert abs(r["x_disp_m"]) < 0.01 and abs(r["y_disp_m"]) < 0.01
    assert 0.077 < r["seat_height_m"] < 0.083
    assert 27.5 < r["reward_per_step"]["total"] < 28.2
    assert r["saturation"] == {"unit": 0.0, "joint": 0.0}
    assert r["steps"] == 340 and r["num_envs"] == NUM_ENVS


def test_rollout_refuses_truncation_length(envs):
    from chair_rl import evaluate

    raw, env = envs
    with pytest.raises(ValueError, match="MAX_EPISODE_LEN"):
        evaluate.rollout(raw, env, evaluate.constant_policy(mdp.A_STAND), steps=mdp.MAX_EPISODE_LEN)


def test_permuted_identity_order_is_transparent(envs):
    """tree 순서·부호 반전 없음이면 permuted 는 항등이어야 한다."""
    import torch

    from chair_rl import evaluate
    from chair_rl import obs_layout as ol

    raw, env = envs
    with torch.inference_mode():
        raw.reset()
        obs = env.get_observations()
    seen = {}
    def probe(o):
        seen["o"] = o.clone()
        return torch.arange(6.0, device=o.device).expand(o.shape[0], -1)
    out = evaluate.permuted(probe, ol.POLICY_JOINT_NAMES)(obs)
    torch.testing.assert_close(seen["o"], obs["policy"])
    torch.testing.assert_close(out[0], torch.arange(6.0, device=out.device))
```

- [ ] **Step 8: Kit 테스트 실행**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_eval_harness.py --isaac -q`
Expected: `3 passed` (종료 시 SIGSEGV 로 exit 139 가 날 수 있다 — `test_env_smoke.py` docstring. 요약이 `3 passed` 면 통과다.)

- [ ] **Step 9: `scripts/eval.py` CLI**

`isaac/scripts/eval.py`:

```python
"""정책 하나를 Chair-Walk-Direct-v0 에서 돌려 §7 지표를 JSON 으로 (스펙 #18 §4, 상위 스펙 §7 평가 하네스).

    PYTHONPATH= python -u isaac/scripts/eval.py --headless --num_envs 64 --seed 123 \
        (--rsl_rl CKPT.pt | --rl_games CKPT.pth | --onnx models/walk.onnx [--joint_order tree] [--quat_flip] | --a_stand) \
        --out result.json

JSON 은 close() *전에* 쓴다 — Kit 종료 시 SIGSEGV 가 날 수 있어 종료코드를 믿지 않는다.
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chair-Walk-Direct-v0 정책 평가")
parser.add_argument("--task", default="Chair-Walk-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=123, help="학습 시드(42)와 다르게 — 외운 초기조건 배제")
parser.add_argument("--steps", type=int, default=340)
parser.add_argument("--clip_actions", type=float, default=1.0, help="래퍼 클램프. 논문 규약 1.0")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--rsl_rl", help="rsl_rl model_*.pt")
src.add_argument("--rl_games", help="rl_games *.pth")
src.add_argument("--onnx", help="walk.onnx 형 ONNX")
src.add_argument("--a_stand", action="store_true", help="a_stand 고정 (기준선)")
parser.add_argument("--joint_order", default="tree", choices=["tree", "actuator", "tree-reversed"],
                    help="--onnx/--rl_games 정책의 출력 순서 가정")
parser.add_argument("--quat_flip", action="store_true", help="쿼터니언 이력 x, y 부호 반전 (실기 IMU 규약)")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata as md  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import chair_rl  # noqa: E402, F401
from chair_rl import evaluate, mdp  # noqa: E402
from chair_rl.mlp_policy import MlpPolicy  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
raw = gym.make(args.task, cfg=env_cfg).unwrapped
env = RslRlVecEnvWrapper(raw, clip_actions=args.clip_actions)
sigma = None

if args.a_stand:
    act, source = evaluate.constant_policy(mdp.A_STAND), "a_stand"
elif args.rsl_rl:
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                             md.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.rsl_rl)
    policy = runner.get_inference_policy(device=raw.device)
    with torch.inference_mode():                     # σ 는 분포가 한 번 update 된 뒤에야 읽힌다
        policy(env.get_observations(), stochastic_output=True)
        sigma = policy.output_std[0].tolist()
    act, source = (lambda obs: policy(obs)), args.rsl_rl
else:
    model = (MlpPolicy.from_rl_games if args.rl_games else MlpPolicy.from_onnx)(args.rl_games or args.onnx, device=raw.device)
    sigma = None if model.sigma is None else model.sigma.tolist()
    act = evaluate.permuted(model, evaluate.JOINT_ORDERS[args.joint_order], args.quat_flip)
    source = args.rl_games or args.onnx

result = evaluate.rollout(raw, env, act, steps=args.steps, clip_actions=args.clip_actions)
result.update({"source": source, "seed": args.seed, "joint_order": args.joint_order,
               "quat_flip": args.quat_flip, "sigma": sigma})
with open(args.out, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps({k: result[k] for k in ("source", "completion_rate", "forward_speed_mps", "seat_height_m")}, ensure_ascii=False))
sys.stdout.flush()

raw.close()
raw.sim.stop()
app.close()
```

- [ ] **Step 10: CLI 를 `a_stand` 와 `walk.onnx` 로 돌려 확인**

Run:
```bash
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --a_stand --out /tmp/claude-1000/-home-tonnonssi-SOTA/eval_a_stand.json
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --onnx models/walk.onnx --out /tmp/claude-1000/-home-tonnonssi-SOTA/eval_onnx.json
```
Expected: 첫 번째 JSON `completion_rate 1.0`, `seat_height_m ≈ 0.080`, `reward_per_step.total ≈ 27.96`. 두 번째 `completion_rate 0.0`(리포트 §7: 논문 정책은 우리 시뮬에서 전도), `sigma` 평균 ≈ 0.066. 종료코드는 보지 않는다.

- [ ] **Step 11: 커밋**

```bash
git add isaac/chair_rl/evaluate.py isaac/scripts/eval.py isaac/tests/test_eval_harness.py
git commit -m "feat[isaac]: eval.py — §7 평가 하네스, 리포트 §9 의 함정을 코드로 고정 (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 12: 전체 CPU 테스트 + PR**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac -q` → 기존 47 + 2 passed.
`git diff --stat feat/15-train-path` 가 400 을 넘으면 멈추고 보고. PR: base `feat/15-train-path`, 본문에 "이 코드가 틀렸다면": (a) `reset_terminated` 가 그 스텝의 종료를 정확히 반영한다는 가정 — `DirectRLEnv.step` 의 `_get_dones → _reset_idx` 순서에 의존; (b) rl_games 키 이름은 합성 테스트만 통과했고 실제 `.pth` 는 Task 4 스모크에서 처음 읽는다; (c) `permuted` 의 inv/perm 방향.

---

### Task 2 (A1): 액션 규약 — 래퍼 [−1, 1] 클램프

브랜치 `feat/18-action-clamp` (Task 1 브랜치 위).

**Files:**
- Modify: `isaac/chair_rl/agents/rsl_rl_ppo_cfg.py` (마지막 블록 `clip_actions = None` 과 그 주석 4줄)
- Modify: `isaac/tests/test_agent_cfg.py:39-41` (`test_actions_are_not_clipped_by_the_wrapper`)
- Modify: `isaac/chair_rl/obs_layout.py` docstring 항목 2 (3줄)
- Modify: `docs/specs/2026-08-25-isaac-rl-design.md` §3 "### 행동" 문단 뒤

**Interfaces:**
- Produces: `WalkPPORunnerCfg.clip_actions == 1.0`. Task 3·4·5 가 이 값을 전제한다.

- [ ] **Step 1: 테스트를 뒤집는다**

`isaac/tests/test_agent_cfg.py` 의 `test_actions_are_not_clipped_by_the_wrapper` 를 통째로 교체:

```python
def test_actions_clamped_to_unit_box(cfg):
    """논문(rl_games clip_actions) 규약: 래퍼가 mu 를 [-1, 1] 로 클램프한 값이 이력·비용에 들어간다.
    관절 목표 ±0.873 클립은 env 가 따로 한다. 실기(rl_walk.py:95)는 raw 를 넣지만 bounds loss 가
    |mu| ≤ 1.1 로 묶으므로 차이 ≤ 0.1 rad (스펙 #18 §1)."""
    assert cfg.clip_actions == 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_agent_cfg.py --isaac -q -k clamped`
Expected: `1 failed` — `assert None == 1.0`

- [ ] **Step 3: cfg 수정**

`isaac/chair_rl/agents/rsl_rl_ppo_cfg.py` 끝의 주석 4줄 + `clip_actions = None` 을 교체:

```python
    # 논문 규약 (스펙 #18 §1): rl_games 는 clip_actions=True 기본으로 mu 를 [-1,1] 로 클램프한 뒤
    # env 에 넘긴다 — env 는 그 값을 이력에 넣고 관절은 ±0.873 으로 한 번 더 클립한다. 실기의
    # ACT_INIT=1.0 은 이 공간의 상한이다. #15 는 실기(rl_walk.py:95)가 raw mu 를 넣는다는 이유로
    # None 이었지만, raw 는 mu 폭주(리포트 2026-08-28 §4)를 관측에 그대로 흘려 논문 학습 분포와
    # 달라진다. bounds loss(A3)가 |mu| ≤ 1.1 로 묶으므로 실기와의 차이는 최대 0.1 rad.
    clip_actions = 1.0
```

- [ ] **Step 4: 통과 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_agent_cfg.py --isaac -q`
Expected: `5 passed`

- [ ] **Step 5: 계약 문구 — `obs_layout.py` docstring 항목 2 교체**

기존 3줄("2. 이력에 들어가는 액션은 **클립 전 정책 출력**…" 부터 "(ACT_INIT = 1.0 이 관절 한계 밖인 것도 이력이 raw 값이라는 정황이다.)" 까지)을:

```
2. 이력에 들어가는 액션은 래퍼가 **[-1, 1] 로 클램프한 정책 출력**이다 (논문 rl_games 규약,
   스펙 #18 §1). ACTION_LIMIT(±0.873) 클립은 관절 목표에만 적용한다. 실기(rl_walk.py)는 raw mu
   를 넣지만 학습이 bounds loss 로 |mu| ≤ 1.1 을 강제하므로 차이는 ≤ 0.1 rad. ACT_INIT = 1.0
   은 이 공간의 상한값이다.
```

- [ ] **Step 6: 스펙 §3 "### 행동" 문단 뒤에 블록 추가**

`docs/specs/2026-08-25-isaac-rl-design.md` 의 "절대 목표각(증분 아님), rad, `±0.872665`(=±50°)로 클립. 10 Hz 제어(물리 1/120 →\ndecimation 12)." 바로 뒤에:

```
> **2026-08-28 개정 (이슈 #18).** 정책 출력은 래퍼에서 `[−1, 1]` 로 클램프되고 그 값이 관측
> 이력에 들어간다(rl_games `clip_actions` 규약). 관절 목표의 `±0.872665` 클립은 env 가 따로
> 한다. 실기 `rl_walk.py` 는 raw mu 를 이력에 넣지만 bounds loss(`|mu| ≤ 1.1`)가 차이를
> 0.1 rad 이하로 묶는다. 근거와 실기 계약 차이는 `docs/specs/2026-08-28-walk-paper-repro.md` §1.
```

- [ ] **Step 7: CPU 테스트 + 커밋**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac -q` → 전부 passed.

```bash
git add isaac/chair_rl/agents/rsl_rl_ppo_cfg.py isaac/tests/test_agent_cfg.py isaac/chair_rl/obs_layout.py docs/specs/2026-08-25-isaac-rl-design.md
git commit -m "feat[isaac]: 액션 규약 — 래퍼가 mu 를 [-1,1] 로 클램프, 이력은 그 값 (논문 rl_games 규약) (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

PR 본문 "틀렸다면": 논문 이력이 클램프값이었다는 것은 `ACT_INIT=1.0` 정황뿐이다; 틀렸다면 실기 이식 때 이력 분포가 어긋나는 쪽은 우리이고 차이는 bounds loss 가 묶는 0.1 rad 이내.

---

### Task 3 (A3): rsl_rl 팔 — `BoundedPPO`

브랜치 `feat/18-bounded-ppo` (Task 2 브랜치 위).

**Files:**
- Create: `isaac/chair_rl/agents/bounded_ppo.py`
- Modify: `isaac/chair_rl/agents/rsl_rl_ppo_cfg.py` (`algorithm = RslRlPpoAlgorithmCfg(...)` 블록)
- Test: `isaac/tests/test_bounded_ppo.py` (CPU), `isaac/tests/test_agent_cfg.py` (확장)

**Interfaces:**
- Produces `bounded_ppo.bound_loss(mu, soft_bound=1.1) -> Tensor(scalar)`, `bounded_ppo.attach_bound_loss(actor, coef, soft_bound=1.1) -> dict(state)`, `bounded_ppo.BoundedPPO(PPO)` (kwarg `bounds_loss_coef`), `rsl_rl_ppo_cfg.BoundedPpoAlgorithmCfg`.
- 텐서보드 키 `Loss/bound`.

**메커니즘 (왜 optimizer.step 훅이 아닌가).** `PPO.update()` 는 `loss.backward()` 로 그래프를 해제한 뒤 `optimizer.step()` 을 부른다. step 직전에 `bound_loss.backward()` 를 다시 부르면 "backward through the graph a second time" 으로 죽는다. 대신 actor 의 forward 훅에서 `output_mean`(= `mlp_output` 텐서 그 자체, `GaussianDistribution.update` 가 `Normal(mean=mlp_output, …)` 로 만든다) 에 **텐서 gradient 훅**을 건다: `∂(coef·bound_loss)/∂mu = coef·2·(clamp_min(mu−1.1,0) + clamp_max(mu+1.1,0)) / B` 를 흘러오는 gradient 에 더한다. 같은 backward 안에서 더해지므로 `clip_grad_norm_` 이 총합에 걸린다 — rl_games 와 정확히 같은 의미론. 스펙 결정 4 의 "optimizer.step 훅" 은 이 계획에서 이렇게 정정한다.

- [ ] **Step 1: 실패하는 CPU 테스트**

`isaac/tests/test_bounded_ppo.py`:

```python
"""rl_games bound_loss 이식이 해석적 gradient 와 맞는지. rsl_rl 은 Kit 없이 import 된다."""

import torch
from torch import nn

from chair_rl.agents.bounded_ppo import SOFT_BOUND, attach_bound_loss, bound_loss


class FakeActor(nn.Module):
    """MLPModel 흉내: forward 뒤 output_mean 이 그래프 안의 mu 텐서다."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2, bias=False)
        self._mean = None

    def forward(self, x, **kwargs):
        self._mean = self.lin(x)
        return self._mean

    @property
    def output_mean(self):
        return self._mean


def test_bound_loss_matches_rl_games_formula():
    mu = torch.tensor([[0.5, -0.5], [2.0, -3.0]])
    # 행별 sum: [0, (0.9)^2 + (1.9)^2] → 배치 평균
    want = ((2.0 - SOFT_BOUND) ** 2 + (3.0 - SOFT_BOUND) ** 2) / 2
    assert torch.isclose(bound_loss(mu), torch.tensor(want))


def test_hook_adds_analytic_gradient_only_outside_bound():
    torch.manual_seed(0)
    actor = FakeActor()
    with torch.no_grad():
        actor.lin.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))   # mu = x[:, :2]
    coef = 0.5
    state = attach_bound_loss(actor, coef)
    x = torch.tensor([[2.0, -0.5, 1.0], [0.3, -2.0, 1.0]])                        # mu = [[2, -.5], [.3, -2]]
    mu = actor(x)
    (mu.sum() * 0.0).backward()          # 주 손실 0 → 남는 gradient 는 훅이 더한 것뿐
    B = x.shape[0]
    excess = torch.tensor([[2.0 - SOFT_BOUND, 0.0], [0.0, -2.0 + SOFT_BOUND]])     # clamp_min/clamp_max
    d_mu = coef * 2.0 * excess / B
    want = d_mu.T @ x                     # ∂/∂W of (d_mu · (W x))
    torch.testing.assert_close(actor.lin.weight.grad, want)
    assert abs(state["last"] - bound_loss(mu.detach()).item()) < 1e-6


def test_hook_is_inert_inside_bound_and_without_grad():
    actor = FakeActor()
    state = attach_bound_loss(actor, 1.0)
    x = torch.tensor([[0.2, -0.7, 0.0]])
    mu = actor(x)
    (mu.sum() * 0.0).backward()
    assert torch.count_nonzero(actor.lin.weight.grad) == 0
    with torch.no_grad():                # 롤아웃 경로 — 훅이 아무것도 하지 않아야 한다
        actor(x)
    assert state["n"] == 1
```

- [ ] **Step 2: 실패 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_bounded_ppo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'chair_rl.agents.bounded_ppo'`

- [ ] **Step 3: 구현**

`isaac/chair_rl/agents/bounded_ppo.py`:

```python
"""rl_games 의 bound_loss 를 rsl_rl PPO 에 이식 (스펙 #18 §3).

rl_games a2c_continuous.bound_loss:
    soft_bound = 1.1
    b = (clamp_min(mu − 1.1, 0)² + clamp_max(mu + 1.1, 0)²).sum(−1)   → 배치 평균 × bounds_loss_coef
mu 가 [−1.1, 1.1] 을 벗어난 만큼만 벌한다. 리포트 2026-08-28 §4 의 mu 폭주(2900°)를 막는 유일한 항이다.

rsl_rl PPO.update() 는 loss.backward() 로 그래프를 해제한 뒤 optimizer.step() 을 부르므로
그 사이에 두 번째 backward 를 넣을 수 없다. 대신 actor 의 forward 훅에서 output_mean 텐서에
gradient 훅을 걸어 ∂(coef·b)/∂mu 를 같은 backward 에 더한다 — clip_grad_norm_ 이 총합에
걸리므로 rl_games 와 의미론이 같다. output_mean 은 GaussianDistribution.update 가
Normal(mean=mlp_output) 으로 만든 그 텐서다 (rsl_rl 5.0.1 distribution.py:169).
rsl_rl 과 torch 만 의존한다. Kit 없음.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from torch import nn

SOFT_BOUND = 1.1


def bound_loss(mu: torch.Tensor, soft_bound: float = SOFT_BOUND) -> torch.Tensor:
    high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
    low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
    return (high + low).sum(-1).mean()


def attach_bound_loss(actor: nn.Module, coef: float, soft_bound: float = SOFT_BOUND) -> dict:
    """actor.forward 뒤 output_mean 에 gradient 훅을 건다. 반환 state: last(마지막 값), sum, n."""
    state = {"last": 0.0, "sum": 0.0, "n": 0}

    def on_forward(module: nn.Module, inputs, output) -> None:
        mu = module.output_mean
        if mu is None or not mu.requires_grad:
            return                                       # 롤아웃(no_grad/inference) 경로
        excess = torch.clamp_min(mu - soft_bound, 0.0) + torch.clamp_max(mu + soft_bound, 0.0)
        d_mu = (coef * 2.0 / mu.shape[0]) * excess.detach()
        mu.register_hook(lambda grad: grad + d_mu)
        b = bound_loss(mu.detach(), soft_bound).item()
        state["last"], state["sum"], state["n"] = b, state["sum"] + b, state["n"] + 1

    actor.register_forward_hook(on_forward)
    return state


class BoundedPPO(PPO):
    """cfg: algorithm.class_name = "chair_rl.agents.bounded_ppo:BoundedPPO", bounds_loss_coef."""

    def __init__(self, *args, bounds_loss_coef: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds_loss_coef = bounds_loss_coef
        self._bound = attach_bound_loss(self.actor, bounds_loss_coef)

    def update(self) -> dict[str, float]:
        self._bound["sum"], self._bound["n"] = 0.0, 0
        loss_dict = super().update()
        loss_dict["bound"] = self._bound["sum"] / max(self._bound["n"], 1)   # 텐서보드 Loss/bound
        return loss_dict
```

- [ ] **Step 4: 통과 확인**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_bounded_ppo.py -q`
Expected: `3 passed`

- [ ] **Step 5: cfg — `BoundedPpoAlgorithmCfg` 와 Ant 매핑**

`isaac/chair_rl/agents/rsl_rl_ppo_cfg.py`: import 에 `RslRlPpoAlgorithmCfg` 는 그대로 두고 클래스 위에 추가:

```python
@configclass
class BoundedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """rl_games bound_loss 이식 (agents/bounded_ppo.py). rsl_rl resolve_callable 이 module:Class 를 받는다."""
    class_name: str = "chair_rl.agents.bounded_ppo:BoundedPPO"
    bounds_loss_coef: float = 1e-4        # [기본] IsaacGymEnvs Ant·Humanoid PPO yaml 공통
```

그리고 `algorithm = RslRlPpoAlgorithmCfg(` 블록을 다음으로 교체 (바뀐 셋: 클래스, `num_learning_epochs` 5→4, `desired_kl` 0.01→0.008 — rl_games 팔(A2)의 Ant yaml 과 같은 값):

```python
    # [기본] IsaacGymEnvs Ant PPO yaml 과 같은 값 (A2 의 rl_games_ppo_cfg.yaml 과 1:1):
    #   lr 3e-4 adaptive kl 0.008, gamma .99, tau .95, e_clip .2, entropy 0, mini_epochs 4,
    #   minibatch 32768 (= 4096 × 32 / 4), critic_coef 2 ↔ value_loss_coef 1.0 (rl_games 는 0.5·critic_coef).
    algorithm = BoundedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
```

- [ ] **Step 6: Kit cfg 테스트 확장** — `isaac/tests/test_agent_cfg.py` 끝에 추가:

```python
def test_algorithm_is_bounded_ppo_with_ant_values(cfg):
    """rl_games 팔과 같은 값이어야 두 팔의 차이가 스택뿐이다 (스펙 #18 §3)."""
    assert cfg.algorithm.class_name == "chair_rl.agents.bounded_ppo:BoundedPPO"
    assert cfg.algorithm.bounds_loss_coef == 1e-4
    assert cfg.algorithm.num_learning_epochs == 4
    assert cfg.algorithm.desired_kl == 0.008
    assert cfg.algorithm.num_mini_batches == 4 and cfg.num_steps_per_env == 32
```

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_agent_cfg.py --isaac -q` → `6 passed`

- [ ] **Step 7: 2-iteration 학습 스모크 — 훅이 실제 러너에서 산다**

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/rsl_rl/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 64 --max_iterations 2 2>&1 | grep -E "Learning iteration|bound|Traceback"`
Expected: `Learning iteration 0/2`, `1/2` 두 줄. 텐서보드 `Loss/bound` 확인: `~/miniforge3/envs/env_isaaclab/bin/python -c "from tensorboard.backend.event_processing import event_accumulator as ea; import glob; a=ea.EventAccumulator(sorted(glob.glob('logs/rsl_rl/chair_walk/*/'))[-1]); a.Reload(); print([t for t in a.Tags()['scalars'] if 'bound' in t])"` → `['Loss/bound']`. Traceback 이 있으면 멈추고 보고.

- [ ] **Step 8: 커밋 + PR**

```bash
git add isaac/chair_rl/agents/bounded_ppo.py isaac/chair_rl/agents/rsl_rl_ppo_cfg.py isaac/tests/test_bounded_ppo.py isaac/tests/test_agent_cfg.py
git commit -m "feat[isaac]: BoundedPPO — rl_games bound_loss 를 rsl_rl 에 텐서 gradient 훅으로 이식 (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

PR "틀렸다면": (a) `output_mean` 이 `mlp_output` 과 같은 텐서 객체가 아니면(예: 미래 rsl_rl 이 `.clone()` 하면) 훅이 그래프 밖에 걸려 조용히 무효 — `test_hook_adds_analytic_gradient` 는 FakeActor 라 잡지 못하고, Step 7 의 `Loss/bound` 가 0 이 아니어도 gradient 는 안 들어갈 수 있다. A4 의 포화율 곡선이 최종 검증. (b) 훅이 minibatch 마다 새로 등록되고 mu 텐서가 해제되며 같이 사라진다 — 누적 안 됨을 확인한 근거는 `register_hook` 이 텐서에 붙는다는 것뿐.

---

### Task 4 (A2): rl_games 팔

브랜치 `feat/18-rl-games` (Task 3 브랜치 위).

**Files:**
- Create: `isaac/scripts/_upstream.py`, `isaac/scripts/rl_games/train.py`, `isaac/chair_rl/agents/rl_games_ppo_cfg.yaml`
- Modify: `isaac/scripts/rsl_rl/train.py` (본문을 shim 으로), `isaac/chair_rl/__init__.py` (kwargs), `isaac/pyproject.toml` (package-data), `isaac/tests/test_registry.py`

**Interfaces:**
- Produces `_upstream.run_upstream(subdir: str, script: str = "train.py") -> None`.
- 체크포인트 경로 `logs/rl_games/chair_walk/<ts>/nn/{chair_walk.pth (best), last_chair_walk_ep_<N>_rew_<R>.pth}` — Task 5 가 `--rl_games` 에 넘긴다.

- [ ] **Step 1: 등록 테스트 추가 (실패 먼저)** — `isaac/tests/test_registry.py` 끝에:

```python
def test_rl_games_entry_point_is_a_yaml_string():
    """rl_games cfg 는 module:file.yaml 문자열 — load_cfg_from_registry 가 파일로 읽는다. import 없음."""
    spec = gym.spec("Chair-Walk-Direct-v0")
    assert spec.kwargs["rl_games_cfg_entry_point"] == "chair_rl.agents:rl_games_ppo_cfg.yaml"
    assert "chair_rl.agents.rsl_rl_ppo_cfg" not in sys.modules
```

Run: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac/tests/test_registry.py -q` → `1 failed` (KeyError)

- [ ] **Step 2: 등록 + package-data**

`isaac/chair_rl/__init__.py` kwargs 에 한 줄:
```python
        "rl_games_cfg_entry_point": "chair_rl.agents:rl_games_ppo_cfg.yaml",
```
docstring 의 "agent cfg 도 문자열이다 …" 문장 뒤에 " rl_games cfg 는 yaml 파일이라 import 자체가 없다 (이슈 #18)." 추가.

`isaac/pyproject.toml` `[tool.setuptools]` 블록 뒤에:
```toml
[tool.setuptools.package-data]
"chair_rl.agents" = ["*.yaml"]
```

Run 테스트 → `3 passed`.

- [ ] **Step 3: rl_games yaml**

`isaac/chair_rl/agents/rl_games_ppo_cfg.yaml`:

```yaml
# Chair-Walk-Direct-v0 rl_games PPO cfg (스펙 #18 §2). 뼈대 = Isaac Lab direct/ant/agents/rl_games_ppo_cfg.yaml.
# 값 출처: [측정] models/walk.onnx / [기본] IsaacGymEnvs Ant PPO yaml / [가정] 논문 미상.
params:
  seed: 42

  env:
    clip_actions: 1.0        # 결정 2 — 래퍼가 mu 를 [-1,1] 로 클램프, env 는 그 값을 이력에
    clip_observations: 5.0   # [측정] walk.onnx 의 Clip(-5, 5) (= rl_games RunningMeanStd 기본)

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0              # [측정] exp(0) = 1 — 상태무관 sigma[6]
        fixed_sigma: True     # [측정]
    mlp:
      units: [1024, 512]      # [측정] actor_mlp.0.weight [1024, 40], .2.weight [512, 1024]
      activation: elu         # [측정]
      d2rl: False
      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: False
  load_path: ''

  config:
    name: chair_walk
    env_name: rlgpu
    device: 'cuda:0'
    device_name: 'cuda:0'
    multi_gpu: False
    ppo: True
    mixed_precision: True     # [기본] Ant
    normalize_input: True     # [측정] RunningMeanStd 가 ONNX 그래프 선두에 융합
    normalize_value: True     # [기본] Ant
    value_bootstrap: True     # [기본] Ant
    num_actors: -1            # train.py 가 --num_envs 로 채운다
    reward_shaper:
      scale_value: 1.0        # 논문 미상. normalize_advantage/normalize_value 아래서 불활성 — 로그가 Table IV 산술과 맞도록 1.0
    normalize_advantage: True
    gamma: 0.99               # [기본]
    tau: 0.95                 # [기본]
    learning_rate: 3e-4       # [기본]
    lr_schedule: adaptive
    schedule_type: legacy
    kl_threshold: 0.008       # [기본] Ant
    score_to_win: 1000000     # 조기 종료 방지 (Ant 20000; 우리 에피소드 리턴은 1 만 이하)
    max_epochs: 1500          # [가정] 4096 × 32 × 1500 = 196.6 M env-step (#15 와 동일). --max_iterations 로 덮임
    save_best_after: 100
    save_frequency: 50
    grad_norm: 1.0
    entropy_coef: 0.0         # [기본] (= 상위 스펙 §7)
    truncate_grads: True
    e_clip: 0.2               # [기본]
    horizon_length: 32        # [가정] 상위 스펙 §7 결정 (Ant 16 / Humanoid 32)
    minibatch_size: 32768     # [기본] Ant (= 4096 × 32 / 4)
    mini_epochs: 4            # [기본] Ant
    critic_coef: 2            # [기본] Ant
    clip_value: True
    seq_length: 4
    bounds_loss_coef: 0.0001  # [기본] Ant·Humanoid 공통 — 이 이슈의 핵심 (리포트 2026-08-28 §4)
```

- [ ] **Step 4: `_upstream.py` 로 shim 공통부 추출**

`isaac/scripts/_upstream.py`:

```python
"""Isaac Lab upstream 학습 스크립트를 그 자리에서 실행하는 shim 의 공통부 (이슈 #15, #18).

설계문서 §1 은 train.py 사본을 말하지만 복사하지 않는다 — 사본은 cli_args 까지 ~350 줄이고
실질 저작은 `import chair_rl` 한 줄뿐이며 upstream 이 움직이면 조용히 썩는다 (#15 결정).

주의: hydra 는 main() 의 예외를 삼키고 "Error executing job" 만 찍은 뒤 exit 0 을 낸다.
학습이 돌았는지는 종료코드가 아니라 로그의 "Learning iteration"(rsl_rl) / "epoch:"(rl_games) 로 본다.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import runpy
import sys


def isaaclab_root() -> pathlib.Path:
    """upstream 저장소 루트. isaaclab.sh 가 export 하는 ISAACLAB_PATH 를 먼저 본다."""
    candidates = []
    env_path = os.environ.get("ISAACLAB_PATH")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    spec = importlib.util.find_spec("isaaclab")           # import 하지 않는다 — 경로만
    if spec is not None and spec.origin is not None:      # <root>/source/isaaclab/isaaclab/__init__.py
        candidates.append(pathlib.Path(spec.origin).resolve().parents[3])
    for root in candidates:
        if (root / "scripts" / "reinforcement_learning").is_dir():
            return root
    raise SystemExit(
        "Isaac Lab 저장소를 찾지 못했다. `source ~/IsaacLab/isaaclab.sh` 로 ISAACLAB_PATH 를 잡거나\n"
        f"ISAACLAB_PATH=<IsaacLab 루트> 를 직접 준다. 찾아본 곳: {[str(c) for c in candidates]}"
    )


def run_upstream(subdir: str, script: str = "train.py") -> None:
    """scripts/reinforcement_learning/<subdir>/<script> 를 chair_rl 등록 뒤 __main__ 으로 실행."""
    upstream = isaaclab_root() / "scripts" / "reinforcement_learning" / subdir
    if not (upstream / script).is_file():
        raise SystemExit(f"upstream 스크립트 없음: {upstream / script}")
    sys.path.insert(0, str(upstream))                     # upstream 의 `import cli_args` 는 같은 디렉터리 전제
    import chair_rl  # noqa: F401  — gym.register 부수효과. gym.make 전에 있어야 한다.

    runpy.run_path(str(upstream / script), run_name="__main__")
```

`isaac/scripts/rsl_rl/train.py` 전체를 다음으로 교체 (기존 docstring 의 실행 예시·주의는 `_upstream.py` 로 옮겼다):

```python
"""rsl_rl 학습 shim (#15). 공통부는 ../_upstream.py.

    PYTHONPATH= python -u isaac/scripts/rsl_rl/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 4096
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _upstream import run_upstream  # noqa: E402

run_upstream("rsl_rl")
```

`isaac/scripts/rl_games/train.py`:

```python
"""rl_games 학습 shim (#18). 공통부는 ../_upstream.py.

    PYTHONPATH= python -u isaac/scripts/rl_games/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 4096
체크포인트: logs/rl_games/chair_walk/<ts>/nn/{chair_walk.pth (best), last_chair_walk_ep_<N>_rew_<R>.pth}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _upstream import run_upstream  # noqa: E402

run_upstream("rl_games")
```

rsl_rl shim 회귀 확인: `PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/rsl_rl/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 64 --max_iterations 1 2>&1 | grep -c "Learning iteration"` → `1`.

- [ ] **Step 5: rl_games 설치**

```bash
mkdir -p logs && ~/miniforge3/envs/env_isaaclab/bin/python -m pip freeze > logs/pip-before-rl_games.txt
~/miniforge3/envs/env_isaaclab/bin/python -m pip install -e "~/IsaacLab/source/isaaclab_rl[rl-games]"
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -c "import rl_games, rsl_rl; print(rl_games.__file__)"
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -m pytest isaac -q
```
Expected: import 성공, CPU 테스트 전부 통과(설치가 torch/tensordict 를 갈아엎지 않았는지). 설치 실패나 테스트 회귀면 **멈추고 보고(선행조건)** — `pip freeze` 차이를 첨부.

- [ ] **Step 6: rl_games 2-iteration 스모크 + 체크포인트를 `eval.py` 로 읽기**

```bash
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/rl_games/train.py --task Chair-Walk-Direct-v0 --headless --num_envs 64 --max_iterations 2 2>&1 | grep -E "epoch:|Actions clipping|Traceback|Error executing"
CK=$(ls -t logs/rl_games/chair_walk/*/nn/last_*.pth | head -1); echo $CK
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --rl_games "$CK" --num_envs 16 --out /tmp/claude-1000/-home-tonnonssi-SOTA/eval_rlg_smoke.json && head -20 /tmp/claude-1000/-home-tonnonssi-SOTA/eval_rlg_smoke.json
```
Expected: `Actions clipping     : 1.0` (래퍼 액션 공간 = Box(−1, 1) → rescale 항등, 스펙 §9 두 번째 항목), `epoch: 1/2`, `epoch: 2/2`, `last_chair_walk_ep_2_rew_*.pth` 존재, eval JSON 이 생성되고 `sigma` 6개가 ≈1.0 (σ_init 0 → exp(0)). `MlpPolicy.from_rl_games` 가 KeyError 를 내면 실제 키를 찍어(`python -c "import torch; print(list(torch.load('$CK', weights_only=False)['model'].keys()))"`) `mlp_policy.py` 의 키 이름을 고치고 Task 1 의 합성 테스트도 같이 고친다.

- [ ] **Step 7: 커밋 + PR**

```bash
git add isaac/scripts/_upstream.py isaac/scripts/rsl_rl/train.py isaac/scripts/rl_games/train.py isaac/chair_rl/agents/rl_games_ppo_cfg.yaml isaac/chair_rl/__init__.py isaac/pyproject.toml isaac/tests/test_registry.py
git commit -m "feat[isaac]: rl_games 팔 — Ant PPO yaml 기반 cfg, 엔트리포인트, shim 공통부 (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
`git diff --stat feat/18-bounded-ppo` 400 초과 시 멈추고 분할안(후보: yaml+등록 / shim 리팩터). PR "틀렸다면": rl_games 의 `clip_actions` 는 래퍼(`RlGamesVecEnvWrapper`)가 하는 것이지 rl_games 본체의 `clip_actions` 가 아닐 수 있다 — 둘 다 clamp(−1,1) 이라 결과는 같지만 어느 쪽이 실제로 걸리는지는 `Actions clipping` 로그와 `env.action_space` 로만 확인했다.

---

### Task 5 (A4): 두 팔 전량 학습 + 비교 리포트

브랜치 `docs/18-walk-paper-repro-report` (Task 4 브랜치 위 — 코드가 필요하다).

**Files:**
- Create: `docs/reports/2026-08-XX-walk-paper-repro.md` (실행일로)
- Modify: `docs/specs/2026-08-28-walk-paper-repro.md` 끝에 "결과" 블록 한 문단

- [ ] **Step 1: 두 팔 학습 (각 ≈25 분, `--video` 는 −12 %)**

```bash
PYTHONPATH= nohup ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/rsl_rl/train.py --task Chair-Walk-Direct-v0 --headless --video --video_length 200 --video_interval 2000 --num_envs 4096 --max_iterations 1500 > logs/train_rsl_bounded.log 2>&1
grep -c "Learning iteration" logs/train_rsl_bounded.log      # 1500
PYTHONPATH= nohup ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/rl_games/train.py --task Chair-Walk-Direct-v0 --headless --video --video_length 200 --video_interval 2000 --num_envs 4096 --max_iterations 1500 > logs/train_rl_games.log 2>&1
grep -c "epoch:" logs/train_rl_games.log                       # ≥ 1500
```
순차로 돌린다 (GPU 하나). 둘 다 `Traceback|Error executing` 이 없어야 한다.

- [ ] **Step 2: 같은 자로 평가 (3 + 체크포인트 곡선)**

```bash
R=logs/rsl_rl/chair_walk/$(ls -t logs/rsl_rl/chair_walk | head -1); G=logs/rl_games/chair_walk/$(ls -t logs/rl_games/chair_walk | head -1)
mkdir -p logs/eval
for it in 50 200 400 600 900 1200 1499; do
  PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --rsl_rl $R/model_$it.pt --out logs/eval/rsl_$it.json
done
for ck in $G/nn/chair_walk.pth $(ls $G/nn/last_*.pth); do
  PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --rl_games $ck --out logs/eval/rlg_$(basename $ck .pth).json
done
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --a_stand --out logs/eval/a_stand.json
PYTHONPATH= ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/scripts/eval.py --headless --onnx models/walk.onnx --out logs/eval/paper_tree.json
```
rl_games 는 `save_frequency 50` 이라 중간 체크포인트가 `nn/` 에 `chair_walk_ep_<N>_rew_<R>.pth` 로 남는지 확인하고(남지 않으면 best/last 둘만) 표에 그대로 적는다.

- [ ] **Step 3: 리포트 작성** — `docs/reports/2026-08-XX-walk-paper-repro.md`, 리포트 2026-08-28 의 구조를 따른다:

```markdown
# walk 1단계 논문 기준 재현 결과 — rl_games 팔 vs rsl_rl+bounds 팔 (이슈 #18)

> 날짜. 스펙 docs/specs/2026-08-28-walk-paper-repro.md 의 A4. 두 팔 모두 4096 env × horizon 32 × 1500 (196.6 M env-step).

## 0. 요약  (각 팔: 완주율 / 전진 속도 / 좌면 높이 → §7 합격선 판정, 굳었는지, 포화율 최종값, σ 최종값 — 5줄)
## 1. 학습 곡선 (iter 0/50/200/400/600/900/1200/1499: eplen, reward, σ, Loss/bound(rsl) 또는 bounds loss(rl_games), progress 항)
## 2. eval.py 같은 자 (표: a_stand / rsl_1499 / rlg_best / rlg_last / walk.onnx — 완주율·속도·높이·total/스텝·포화율 unit/joint·σ)
## 3. 포화율 곡선 (체크포인트별 unit·joint 포화율 — 리포트 2026-08-28 §4 의 표와 같은 형식. bounds loss 가 폭주를 막았는가가 이 문서의 첫 질문)
## 4. 두 팔의 차이 (같은 값으로 맞춘 항목 / 남은 차이: 관측 정규화 클립, value 정규화, mixed precision, grad clip 순서)
## 5. 판정 — 리포트 2026-08-28 §6 의 세 후보 중 "스택" 이 닫혔는가. 다음 이슈 후보 (§6 노이즈 / dt / 관절 대응)
## 6. 이 결과가 틀렸다면 어떻게 틀렸을지
## 7. 재현 (명령 전부, 로그·JSON 경로)
```

- [ ] **Step 4: 스펙에 결과 포인터** — `docs/specs/2026-08-28-walk-paper-repro.md` 끝에:

```markdown
## 결과 (YYYY-MM-DD)

두 팔 완주. 요약과 표는 `docs/reports/YYYY-MM-DD-walk-paper-repro.md`. (한 줄 판정: 걷는가 / 굳는가 / 포화율)
```

- [ ] **Step 5: 커밋 + PR (base `feat/18-rl-games` 또는 그것이 머지된 브랜치)**

```bash
git add docs/reports/ docs/specs/2026-08-28-walk-paper-repro.md
git commit -m "docs[isaac]: walk 논문 기준 재현 결과 — 두 팔 비교 리포트 (#18)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
PR "틀렸다면": 리포트 §6 을 그대로. #18 은 이 PR 로 닫는다.

---

## 계획 자체 검토 (작성 시점)

- 스펙 커버리지: §1→Task 2, §2→Task 4, §3→Task 3, §4→Task 1, §5·§6→Task 4 Step 6·Task 5, §7 분할→브랜치, §8 위험→Task 4 Step 5 의 멈춤 조건, §9→각 PR 본문.
- 스펙 결정 4("optimizer.step 훅")는 Task 3 의 메커니즘 설명대로 **텐서 gradient 훅**으로 정정한다 — 그래프 해제 때문에 step 훅은 동작하지 않는다. 스펙 본문도 같은 커밋에서 고친다.
- 타입 일관성: `ActFn` 입력은 래퍼 obs(TensorDict), `permuted` 는 텐서→텐서 모델을 받아 ActFn 을 낸다, `MlpPolicy.forward` 는 텐서→텐서. `rollout(raw, env, act, steps, clip_actions)` 시그니처는 Task 1·5 동일.
