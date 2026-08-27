"""걷기 MDP 의 순수 함수 — 논문(arXiv 2404.05932) Table III/IV 를 그대로 옮겼다.

모든 함수는 (N, …) 배치 텐서를 받아 (N,) 또는 (N,k) 를 돌려준다. self 도 Isaac 도
모른다 — Direct env(이슈 C)는 이 함수를 부르고 가중합만 한다. 쿼터니언은 실기 규약
(x, y, z, w). 위치·자세는 루트(dummy) = 좌면 중심 기준(설계문서 §3).

논문 기호: p 좌면 중심, q 자세, u_prj = (R_q e_z)_z, p_target = [10, 0, 0], dt = 0.1,
a 서보 지령각(6), ω 서보 각속도(6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# --- 논문 상수 (Table IV) ---
P_TARGET = (10.0, 0.0, 0.0)   # m
HEIGHT_REF = 0.08             # height = min{1, p_z / 0.08}
UP_REF = 0.93                 # up = min{1, u_prj / 0.93}
HEADING_REF = 0.8             # heading = min{1, (R_q e_x · dir) / 0.8}
OMEGA_MAX = 10.472            # rad/s (= 600 deg/s)
OMEGA_TOL = 1.0
CONTROL_DT = 0.1          # 논문 dt: 제어 주기. progress 의 potentials 초기화와 스텝이 같은 값을 써야 한다
MAX_EPISODE_LEN = 350     # 논문 Table III "episode exceeds 350"


# ---------------------------------------------------------------- 회전 헬퍼 (xyzw)

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q (N,4) (x,y,z,w) 로 v (N,3) 또는 (3,) 를 회전. src/utils.py 의 quat_rotate 와 동일."""
    if v.dim() == 1:
        v = v.expand(q.shape[0], 3)
    qv, w = q[:, :3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def up_proj(q: torch.Tensor) -> torch.Tensor:
    """(R_q e_z)_z = 1 - 2(x^2 + y^2). 직립 1, 옆으로 누우면 0, 뒤집히면 -1."""
    return 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)


def heading_proj(q: torch.Tensor, root_pos: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    """R_q e_x · (p_target - p) / ||p_target - p||."""
    fwd = quat_rotate(q, q.new_tensor([1.0, 0.0, 0.0]))
    to_t = p_target - root_pos
    to_t = to_t / to_t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return (fwd * to_t).sum(-1)


# ---------------------------------------------------------------- 보상 항 (가중치 미적용)

def potential(root_pos: torch.Tensor, p_target: torch.Tensor, dt: float = CONTROL_DT) -> torch.Tensor:
    """P = -||p_target - p|| / dt."""
    return -(p_target - root_pos).norm(dim=-1) / dt


def progress(potentials: torch.Tensor, root_pos: torch.Tensor,
             p_target: torch.Tensor, dt: float = CONTROL_DT) -> tuple[torch.Tensor, torch.Tensor]:
    """P - P_pre. (보상, 새 potentials). env 가 potentials 를 들고 리셋 때 재초기화한다."""
    new = potential(root_pos, p_target, dt)
    return new - potentials, new


def height_reward(root_pos: torch.Tensor) -> torch.Tensor:
    return (root_pos[:, 2] / HEIGHT_REF).clamp(max=1.0)


def up_reward(up: torch.Tensor) -> torch.Tensor:
    return (up / UP_REF).clamp(max=1.0)


def heading_reward(hp: torch.Tensor) -> torch.Tensor:
    return (hp / HEADING_REF).clamp(max=1.0)


def action_cost(actions: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
    """||a - a_pre||^2 — 크기가 아니라 변화량."""
    return ((actions - prev_actions) ** 2).sum(-1)


def vel_cost(joint_vel: torch.Tensor) -> torch.Tensor:
    """||ω / (ω_max - ω_tol)||^2."""
    return ((joint_vel / (OMEGA_MAX - OMEGA_TOL)) ** 2).sum(-1)


@dataclass(frozen=True)
class WalkRewardWeights:
    """Table IV 가중치."""
    progress: float = 30.0
    height: float = 20.0
    up: float = 5.0
    heading: float = 2.0
    alive: float = 1.0
    death: float = -1.0
    action: float = -2.0
    vel: float = -2.0


def walk_reward_terms(root_pos: torch.Tensor, root_quat: torch.Tensor, potentials: torch.Tensor,
                      actions: torch.Tensor, prev_actions: torch.Tensor, joint_vel: torch.Tensor,
                      dt: float = CONTROL_DT, p_target: torch.Tensor | None = None):
    """6개 항(가중치 미적용)과 새 potentials. alive/death 는 walk_total 이 처리한다."""
    if p_target is None:
        p_target = root_pos.new_tensor(P_TARGET)
    prog, new_pot = progress(potentials, root_pos, p_target, dt)
    terms = {
        "progress": prog,
        "height": height_reward(root_pos),
        "up": up_reward(up_proj(root_quat)),
        "heading": heading_reward(heading_proj(root_quat, root_pos, p_target)),
        "action": action_cost(actions, prev_actions),
        "vel": vel_cost(joint_vel),
    }
    return terms, new_pot


def walk_total(terms: dict, w: WalkRewardWeights, terminated: torch.Tensor) -> torch.Tensor:
    """Σ w·term + alive. 종료 스텝은 death 로 대체한다 — 논문: "This value overwrites the
    previous rewards". truncation(350 스텝)은 여기 들어오지 않는다(§4)."""
    total = (w.progress * terms["progress"] + w.height * terms["height"] + w.up * terms["up"]
             + w.heading * terms["heading"] + w.action * terms["action"] + w.vel * terms["vel"]
             + w.alive)
    return torch.where(terminated, torch.full_like(total, w.death), total)


# ---------------------------------------------------------------- 종료 조건 (Table III)

# 좌면 플레이트 바닥면의 꼭짓점 4개, 좌면 중심(dummy) 기준 m. MuJoCo 로 잰 값 (계획 문서
# "확정된 상수"): x∈[-0.082,+0.076], y∈[-0.077,+0.081], 바닥면 z=-0.011.
SEAT_CORNERS_LOCAL = (
    (-0.082, -0.077, -0.011), (-0.082, 0.081, -0.011),
    (0.076, -0.077, -0.011), (0.076, 0.081, -0.011),
)
TILT_THRESH = 0.7    # ||q - [0,0,0,1]|| > 0.7  (≈ 82°, yaw 포함)
GROUND_Z = 0.005     # 모서리 월드 z < 5 mm 면 "접지"
HEIGHT_MIN = 0.005   # 좌면 중심 z < 5 mm


def seat_corner_heights(root_pos: torch.Tensor, root_quat: torch.Tensor,
                        corners_local=None) -> torch.Tensor:
    """모서리 4개의 월드 z (N,4). 접촉 센서 없이 기하로 계산한다 (§9.4)."""
    src = SEAT_CORNERS_LOCAL if corners_local is None else corners_local
    c = torch.as_tensor(src, dtype=root_pos.dtype, device=root_pos.device)  # (4,3)
    n = root_pos.shape[0]
    q = root_quat.repeat_interleave(4, dim=0)                 # (N*4, 4)
    v = c.repeat(n, 1)                                        # (N*4, 3)
    world = quat_rotate(q, v).view(n, 4, 3) + root_pos[:, None, :]
    return world[..., 2]


def quat_dist_to_identity(q: torch.Tensor) -> torch.Tensor:
    """||q - [0,0,0,1]||. q 와 -q 는 같은 회전이므로 w >= 0 으로 맞춘 뒤 잰다."""
    qc = torch.where(q[:, 3:4] < 0, -q, q)
    ident = q.new_tensor([0.0, 0.0, 0.0, 1.0])
    return (qc - ident).norm(dim=-1)


def walk_terminated(root_pos: torch.Tensor, root_quat: torch.Tensor):
    """Table III 의 리셋 조건 중 시간 초과를 뺀 셋. (terminated, {tilt, ground, height})."""
    reasons = {
        "tilt": quat_dist_to_identity(root_quat) > TILT_THRESH,
        "ground": seat_corner_heights(root_pos, root_quat).min(dim=-1).values < GROUND_Z,
        "height": root_pos[:, 2] < HEIGHT_MIN,
    }
    terminated = reasons["tilt"] | reasons["ground"] | reasons["height"]
    return terminated, reasons


def walk_truncated(episode_len: torch.Tensor, max_len: int = MAX_EPISODE_LEN) -> torch.Tensor:
    """350 스텝 초과는 실패가 아니라 시간 초과다 — terminated 와 섞지 않는다 (§4).

    episode_len 은 Isaac Lab 의 episode_length_buf (스텝 뒤 증가된 값). >= 350 이면
    정확히 350 스텝이다 — Isaac Lab 번들 env 의 max_episode_length - 1 관례(349 스텝)를
    따르지 않는다."""
    return episode_len >= max_len
