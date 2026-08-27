"""Chair-Walk-Direct-v0 스모크 (설계문서 §9.6). Kit 필요:
    pytest isaac/tests/test_env_smoke.py --isaac -v

env 는 모듈당 하나만 띄운다 — DirectRLEnv 는 SimulationContext 가 이미 있으면 거부하고,
test_asset_build 도 같은 세션에서 자기 컨텍스트를 만들고 지우므로 close() 로 확실히 비운다.
"""

import numpy as np
import pytest
import torch

from chair_rl import mdp
from chair_rl import obs_layout as ol

pytestmark = pytest.mark.isaac

NUM_ENVS = 16


@pytest.fixture(scope="module")
def env(kit_app):
    import gymnasium as gym

    import chair_rl  # noqa: F401  — 등록
    from chair_rl.walk_env import WalkEnvCfg

    cfg = WalkEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    wrapped = gym.make("Chair-Walk-Direct-v0", cfg=cfg)
    yield wrapped.unwrapped
    wrapped.close()


def _rand_actions(env, lo=-ol.ACTION_LIMIT, hi=ol.ACTION_LIMIT):
    return torch.rand(env.num_envs, ol.NUM_ACTIONS, device=env.device) * (hi - lo) + lo


def test_timing_matches_mdp_constants(env):
    assert env.step_dt == pytest.approx(mdp.CONTROL_DT)
    assert env.max_episode_length == mdp.MAX_EPISODE_LEN == 350


def test_reset_obs_is_history_init(env):
    obs, _ = env.reset()
    o = obs["policy"]
    assert o.shape == (NUM_ENVS, ol.OBS_DIM)
    rot = o[:, :16].view(NUM_ENVS, 4, 4)
    assert torch.all(rot[..., :3] == 0.0) and torch.all(rot[..., 3] == 1.0)
    assert torch.all(o[:, 16:] == ol.ACT_INIT)
    # potentials 는 env 로컬 좌표로 계산돼야 한다 — 월드 좌표면 env 원점(spacing 0.6 m)만큼 서로 다르다
    pot = env._potentials.cpu().numpy()
    expect = -np.linalg.norm(np.array(mdp.P_TARGET) - np.array([0.0, 0.0, env.cfg.init_height])) / mdp.CONTROL_DT
    np.testing.assert_allclose(pot, expect, atol=0.1)


def test_50_random_steps_finite(env):
    env.reset()
    for _ in range(50):
        obs, rew, terminated, truncated, extras = env.step(_rand_actions(env))
        o = obs["policy"]
        assert o.shape == (NUM_ENVS, ol.OBS_DIM) and torch.isfinite(o).all()
        assert rew.shape == (NUM_ENVS,) and torch.isfinite(rew).all()
        assert terminated.dtype == torch.bool and truncated.dtype == torch.bool
        assert not truncated.any()  # 50 < 350
    log = extras["log"]
    assert {f"rew/{k}" for k in ("progress", "height", "up", "heading", "action", "vel")} <= set(log)
    assert {f"term/{k}" for k in ("tilt", "ground", "height")} <= set(log)
    # 관절이 실제로 움직였다 (a2j 로 지령이 전달됨)
    assert (env.robot.data.joint_pos - env.robot.data.default_joint_pos).abs().max() > 0.1
