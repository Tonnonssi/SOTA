"""obs_layout 이 실기 src/rl_walk.py 의 이력 로직과 같은 관측을 만드는지.

rl_walk.py 는 rospy 를 import 해서 통째로 import 할 수 없다. 이력 갱신 세 줄
(src/rl_walk.py 의 `while` 루프 안, obs -> action -> history 갱신)을 numpy 로
그대로 옮겨 참조 구현으로 쓴다:
    action = run_onnx_model(..., np.concatenate([rotation_history.flatten(), action_history.flatten()], 0))
    rotation_history = np.concatenate([rotation, rotation_history], 0)[:-1, :]
    action_history   = np.concatenate([action,   action_history],   0)[:-1, :]
초기값도 그대로: rotation_history = zeros(4,4) with [:,3]=1, action_history = ones(4,6).
"""

import numpy as np
import torch

from chair_rl import obs_layout as ol


def _reference_step(rot_his_np, act_his_np, quat_np, act_np):
    """src/rl_walk.py 의 갱신 순서: 먼저 관측(갱신 전), 그 다음 이력 갱신."""
    obs = np.concatenate([rot_his_np.flatten(), act_his_np.flatten()], 0)
    rot_his_np = np.concatenate([quat_np[None], rot_his_np], 0)[:-1, :]
    act_his_np = np.concatenate([act_np[None], act_his_np], 0)[:-1, :]
    return obs, rot_his_np, act_his_np


def test_constants():
    assert ol.OBS_DIM == 40 == ol.NUM_ROT_HIS * 4 + ol.NUM_ACT_HIS * ol.NUM_ACTIONS
    assert ol.ROT_INIT == (0.0, 0.0, 0.0, 1.0)
    assert ol.ACT_INIT == 1.0
    assert ol.ACTION_LIMIT == 0.872665
    # chair_sim.JOINT_ORDERS["tree"] 와 같아야 한다 (chair_sim 은 Isaac 을 import 해서 여기서 못 읽는다)
    assert ol.POLICY_JOINT_NAMES == ("joint2", "joint1", "joint4", "joint3", "joint6", "joint5")


def test_new_history_has_reset_values():
    rot, act = ol.new_history(3, "cpu")
    assert rot.shape == (3, 4, 4) and act.shape == (3, 4, 6)
    assert torch.equal(rot[:, :, :3], torch.zeros(3, 4, 3)) and torch.all(rot[:, :, 3] == 1.0)
    assert torch.all(act == 1.0)


def test_wxyz_to_xyzw():
    q = torch.tensor([[0.9, 0.1, 0.2, 0.3]])          # (w, x, y, z)
    assert torch.equal(ol.wxyz_to_xyzw(q), torch.tensor([[0.1, 0.2, 0.3, 0.9]]))


def test_push_matches_rl_walk_for_100_steps():
    rng = np.random.default_rng(0)
    N = 3
    rot, act = ol.new_history(N, "cpu", dtype=torch.float64)
    ref_rot = [np.concatenate([np.zeros((4, 3)), np.ones((4, 1))], 1) for _ in range(N)]
    ref_act = [np.ones((4, 6)) for _ in range(N)]
    for _ in range(100):
        q = rng.normal(size=(N, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        a = rng.uniform(-0.87, 0.87, size=(N, 6))
        # 참조: 관측은 갱신 *전* 이력. obs_layout.push 는 갱신 *후* 관측을 돌려주므로
        # 참조의 다음 스텝 관측과 비교한다.
        for i in range(N):
            _, ref_rot[i], ref_act[i] = _reference_step(ref_rot[i], ref_act[i], q[i], a[i])
        obs = ol.push(rot, act, torch.tensor(q, dtype=torch.float64), torch.tensor(a, dtype=torch.float64))
        for i in range(N):
            ref_obs = np.concatenate([ref_rot[i].flatten(), ref_act[i].flatten()], 0)
            np.testing.assert_allclose(obs[i].numpy(), ref_obs, rtol=0, atol=1e-12)
    assert obs.shape == (N, 40)


def test_obs_layout_newest_first():
    rot, act = ol.new_history(1, "cpu")
    q1 = torch.tensor([[0.1, 0.2, 0.3, 0.9]]); a1 = torch.full((1, 6), 0.5)
    obs = ol.push(rot, act, q1, a1)
    assert torch.equal(obs[0, 0:4], q1[0])            # 최신 쿼터니언이 맨 앞
    assert torch.equal(obs[0, 4:8], torch.tensor([0., 0., 0., 1.]))  # 그 뒤는 리셋값
    assert torch.equal(obs[0, 16:22], a1[0])          # 액션 구간 첫 줄 = 최신 액션
    assert torch.all(obs[0, 22:40] == 1.0)


def test_reset_history_only_touches_given_envs():
    rot, act = ol.new_history(2, "cpu")
    ol.push(rot, act, torch.tensor([[0.1, 0.2, 0.3, 0.9]] * 2), torch.full((2, 6), 0.5))
    ol.reset_history(rot, act, torch.tensor([1]))
    assert torch.all(rot[1] == torch.tensor([0., 0., 0., 1.])) and torch.all(act[1] == 1.0)
    assert torch.equal(rot[0, 0], torch.tensor([0.1, 0.2, 0.3, 0.9])) and torch.all(act[0, 0] == 0.5)


def test_push_skip_mask_keeps_reset_envs_pristine():
    rot, act = ol.new_history(3, "cpu")
    q = torch.tensor([[0.1, 0.2, 0.3, 0.9]] * 3); a = torch.full((3, 6), 0.5)
    ol.push(rot, act, q, a)                                   # 모두 한 번 진행
    skip = torch.tensor([False, True, False])
    ol.reset_history(rot, act, torch.tensor([1]))             # env 1 리셋
    obs = ol.push(rot, act, q * 0.5, a * 0.2, skip_mask=skip)  # env 1 은 건너뜀
    assert torch.equal(obs[1, 0:16], torch.tensor([0., 0., 0., 1.] * 4))   # 리셋값 그대로
    assert torch.all(obs[1, 16:40] == 1.0)
    assert torch.equal(obs[0, 0:4], (q * 0.5)[0]) and torch.equal(obs[2, 16:22], (a * 0.2)[2])


def test_rl_walk_source_still_matches_reference():
    """참조 구현은 src/rl_walk.py 의 세 줄을 옮긴 것이다. 원문이 바뀌면 여기서 걸린다."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(repo, "src", "rl_walk.py"), encoding="utf-8").read()
    for needle in (
        "rotation_history[:,  3] = 1.0",
        "action_history = np.ones([numActionHis, 6])",
        "np.concatenate([rotation, rotation_history], 0)[:-1, :]",
        "np.concatenate([action, action_history], 0)[:-1, :]",
    ):
        assert needle in src, needle
