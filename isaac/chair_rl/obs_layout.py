"""40차원 관측 레이아웃과 이력 버퍼 — 실기 src/rl_walk.py 와의 계약 (설계문서 §3, §9.3).

obs = [쿼터니언 이력 4x4 (x,y,z,w) | 액션 이력 4x6], 최신이 앞, 총 40.

실기(src/rl_walk.py, while 루프)의 순서는
    action  = policy(concat(rot_his.flatten(), act_his.flatten()))   # 갱신 *전* 이력으로 추론
    rot_his = concat([quat_now, rot_his])[:-1]
    act_his = concat([action,   act_his])[:-1]
이다. env 에서는 _get_observations 가 push(quat_now, action) 의 결과를 다음 추론에
넘기므로 "관측 → 행동 → push" 의 순서는 같다. 계약의 세부 세 가지:

1. 리셋 직후 첫 관측은 **리셋값 그대로**다. 실기는 초기 이력으로 첫 추론을 한다.
   DirectRLEnv 는 _reset_idx 뒤에 _get_observations 를 부르므로 env 는 리셋된 env 에
   대해 push 를 건너뛰어야 한다 — push(..., skip_mask=reset_buf).
2. 이력에 들어가는 액션은 **클립 전 정책 출력**(rl_walk 의 mu)이다. 실기의 safeClip 은
   서보 지령에만 걸린다. ACTION_LIMIT 클립은 관절 목표에만 적용하고 이력에는 raw 를 넣는다.
   (ACT_INIT = 1.0 이 관절 한계 밖인 것도 이력이 raw 값이라는 정황이다.)
3. 타이밍은 한 제어주기만큼 다르다. 실기는 a_t 를 publish 한 직후 IMU 를 읽어 a_t 와
   "a_t 가 작용하기 전" 쿼터니언을 짝짓고, 서보는 commands 큐 때문에 a_t 를 두 주기
   뒤에 실행한다. env 는 a_t 가 0.1 s 작용한 뒤의 쿼터니언을 짝짓는다. 이 차이는 여기서
   고치지 않는다 — 이슈 C(env)와 §6(DR, 액션 지연)에서 결정한다.

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
    # (4,) 를 (K,4,4) 에 브로드캐스트: 이력 4 슬롯 전부를 리셋값으로
    rot_his[env_ids] = torch.tensor(ROT_INIT, device=rot_his.device, dtype=rot_his.dtype)
    act_his[env_ids] = ACT_INIT


def flatten(rot_his: torch.Tensor, act_his: torch.Tensor) -> torch.Tensor:
    """(N,4,4),(N,4,6) -> (N,40). 행 우선이라 obs[0:4] 가 최신 쿼터니언이다."""
    return torch.cat([rot_his.flatten(1), act_his.flatten(1)], dim=1)


def push(rot_his: torch.Tensor, act_his: torch.Tensor,
         quat_xyzw: torch.Tensor, action: torch.Tensor,
         skip_mask: torch.Tensor | None = None) -> torch.Tensor:
    """최신 (quat, action) 을 index 0 에 밀어 넣고(제자리) 갱신된 관측 (N,40) 을 돌려준다.

    skip_mask (N,) bool 이 주어지면 True 인 env 는 밀어 넣지 않는다 — 방금 리셋된 env 의
    첫 관측은 실기처럼 리셋값 그대로여야 한다 (아래 모듈 docstring "리셋 직후" 참조).
    """
    new_rot = torch.cat([quat_xyzw[:, None, :], rot_his[:, :-1]], dim=1)
    new_act = torch.cat([action[:, None, :], act_his[:, :-1]], dim=1)
    if skip_mask is not None:
        keep = skip_mask[:, None, None]
        new_rot = torch.where(keep, rot_his, new_rot)
        new_act = torch.where(keep, act_his, new_act)
    rot_his.copy_(new_rot)
    act_his.copy_(new_act)
    return flatten(rot_his, act_his)
