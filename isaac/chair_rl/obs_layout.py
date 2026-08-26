"""40차원 관측 레이아웃과 이력 버퍼 — 실기 src/rl_walk.py 와의 계약 (설계문서 §3, §9.3).

obs = [쿼터니언 이력 4x4 (x,y,z,w) | 액션 이력 4x6], 최신이 앞, 총 40.

실기(src/rl_walk.py, while 루프)의 순서는
    action  = policy(concat(rot_his.flatten(), act_his.flatten()))
    rot_his = concat([quat_now, rot_his])[:-1]
    act_his = concat([action,   act_his])[:-1]
이다. env 에서는 _pre_physics_step 이 action 을 받고 _get_observations 가
push(quat_now, action) 의 결과를 돌려주므로 같은 순서가 된다.

리셋값(쿼터니언 단위, 액션 1.0)은 실기 초기값 그대로다. 1.0 은 관절 한계(0.8727)
밖이지만 학습 때 그 분포였으므로 바꾸지 않는다.

torch 만 의존한다. Isaac 없음.
"""

from __future__ import annotations

import torch

NUM_ROT_HIS = 4
NUM_ACT_HIS = 4
NUM_ACTIONS = 6
OBS_DIM = NUM_ROT_HIS * 4 + NUM_ACT_HIS * NUM_ACTIONS  # 40
ROT_INIT = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w) 단위 쿼터니언
ACT_INIT = 1.0
ACTION_LIMIT = 0.872665  # rad = 50 deg, MJCF ctrlrange

# 정책 출력 인덱스 0..5 -> MJCF 관절 이름. chair_sim.JOINT_ORDERS["tree"] 와 같다.
# 근거: 논문 Table VI 의 a_stand = [-0.1745, 0, -0.1745, 0, 0.1745, 0] 가 이 순서의
# STANDING_SIM 과 일치한다 (설계문서 §4). 실기 서보 번호와의 대응은 실측 전.
POLICY_JOINT_NAMES = ("joint2", "joint1", "joint4", "joint3", "joint6", "joint5")


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Isaac Lab root_quat_w (w,x,y,z) -> 실기/정책 규약 (x,y,z,w)."""
    return q[..., [1, 2, 3, 0]]


def new_history(num_envs: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """리셋값으로 채운 (rot_his (N,4,4), act_his (N,4,6))."""
    rot_his = torch.zeros(num_envs, NUM_ROT_HIS, 4, device=device, dtype=dtype)
    rot_his[..., 3] = 1.0
    act_his = torch.full((num_envs, NUM_ACT_HIS, NUM_ACTIONS), ACT_INIT, device=device, dtype=dtype)
    return rot_his, act_his


def reset_history(rot_his: torch.Tensor, act_his: torch.Tensor, env_ids: torch.Tensor) -> None:
    """env_ids 의 이력만 리셋값으로 되돌린다 (제자리)."""
    rot_his[env_ids] = torch.tensor(ROT_INIT, device=rot_his.device, dtype=rot_his.dtype)
    act_his[env_ids] = ACT_INIT


def flatten(rot_his: torch.Tensor, act_his: torch.Tensor) -> torch.Tensor:
    """(N,4,4),(N,4,6) -> (N,40). 행 우선이라 obs[0:4] 가 최신 쿼터니언이다."""
    return torch.cat([rot_his.flatten(1), act_his.flatten(1)], dim=1)


def push(rot_his: torch.Tensor, act_his: torch.Tensor,
         quat_xyzw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """최신 (quat, action) 을 index 0 에 밀어 넣고(제자리) 갱신된 관측 (N,40) 을 돌려준다."""
    rot_his[:, 1:] = rot_his[:, :-1].clone()
    rot_his[:, 0] = quat_xyzw
    act_his[:, 1:] = act_his[:, :-1].clone()
    act_his[:, 0] = action
    return flatten(rot_his, act_his)
