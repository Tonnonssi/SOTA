"""Chair-Walk-Direct-v0 스모크 (설계문서 §9.6). Kit 필요:
    pytest isaac/tests/test_env_smoke.py --isaac -v

env 는 모듈당 하나만 띄운다 — DirectRLEnv 는 SimulationContext 가 이미 있으면 거부한다.
SimulationContext.clear_instance() 는 콜백만 지우고 타임라인은 멈추지 않는다 — playing 인
채로 남으면 다음에 만들어지는 SimulationContext 가 reset()/step() 에서 "재생될 때까지" 도는
무한 루프에 빠진다(simulation_context.py 의 step()). 그래서 여기서도 test_asset_build.py 도
close()/clear_instance() 뒤에 sim.stop() 을 부른다 — *뒤에* 인 게 중요하다. stop() 은(터미널
실행이 아닐 때) 안에서 app.update() 를 불러 STOP 이벤트를 동기 디스패치하는데,
clear_instance() 가 구독 해제하는 _app_control_on_stop_handle 콜백이 아직 살아 있으면 그
콜백 자체가 "재생될 때까지" render() 를 도는 별개의 무한 루프에 빠진다(대화형 스크립트가
창을 열어 두게 하려는 의도된 동작 — DirectRLEnv 자체의 reset() 은 이걸 피하려고
_disable_app_control_on_stop_handle 을 잠깐 켠다). 그래서 stop() 을 clear_instance() 앞이
아니라 뒤에 부른다.

주의: 이 stop() 은 test_asset_build.py 와 함께 돌 때의 무한 루프(위)는 확실히 없앤다 —
진단으로 확인함(is_playing()=False, is_stopped()=True, app.close() 전). 하지만 이 파일을
단독으로 --isaac -q 로 돌리면 테스트는 다 통과한 뒤 exit 139(SIGSEGV, core dumped) 로 죽는
게 stop() 을 넣기 전/후 동일하게 재현된다 — 물리 device 를 cpu 로 바꿔도 마찬가지였다.
app.close() 호출 자체는 정상 반환하고 그 *뒤* 프로세스 종료 과정에서 죽으므로, 이 크래시는
타임라인 상태가 아니라 다른 원인(가장 유력한 후보: 헤드리스에서도 항상 GPU 인 Kit 렌더러/
RTX 파이프라인의 종료 처리 — 이 env 는 조명·바닥 평면을 스폰하고 render_interval 로 실제로
렌더를 돌리는데, 크래시가 없는 test_asset_build.py 쪽은 렌더 파이프라인을 전혀 쓰지 않는다)
로 보인다. 자세한 진단 근거는 task-2-report.md 참조 — 여기서 더 파지 않는다.
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
    sim = wrapped.unwrapped.sim
    yield wrapped.unwrapped
    # close() 가 clear_instance() 로 _app_control_on_stop_handle 구독을 먼저 해제한다(모듈
    # docstring 참조). DirectRLEnv.close() 는 (create_stage_in_memory 가 아닌 한) stop() 을
    # 부르지 않으므로, 타임라인을 실제로 멈추는 건 여기서 close() *뒤에* 한다 — 콜백이 이미
    # 해제된 뒤라 안전하다.
    wrapped.close()
    sim.stop()


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
