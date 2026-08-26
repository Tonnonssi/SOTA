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
    fwd = quat_rotate(q, torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device))
    to_t = p_target - root_pos
    to_t = to_t / to_t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return (fwd * to_t).sum(-1)


# ---------------------------------------------------------------- 보상 항 (가중치 미적용)

def potential(root_pos: torch.Tensor, p_target: torch.Tensor, dt: float) -> torch.Tensor:
    """P = -||p_target - p|| / dt."""
    return -(p_target - root_pos).norm(dim=-1) / dt


def progress(potentials: torch.Tensor, root_pos: torch.Tensor,
             p_target: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def walk_reward_terms(root_pos, root_quat, potentials, actions, prev_actions, joint_vel,
                      dt: float, p_target: torch.Tensor):
    """6개 항(가중치 미적용)과 새 potentials. alive/death 는 walk_total 이 처리한다."""
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
