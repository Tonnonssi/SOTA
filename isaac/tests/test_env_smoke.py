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


def test_a2j_routes_policy_index_to_named_joint(env):
    """정책 인덱스 k 에만 목표각을 주면 POLICY_JOINT_NAMES[k] 관절만 그 목표를 받는다.
    지령(joint_pos_target)은 물리 없이 정확히 확인되고, 실제 위치는 방향만 본다.

    방향 확인은 로봇을 바닥에서 띄운 채로 한다 — 서 있는 자세(체중 부하)에서 재보면
    joint5 는 effort_limit(0.3) 이 정지 하중을 버티는 데 이미 소진돼 목표를 더 줘도
    실측으로 전혀 안 움직인다(16 env 전부 |Δ|<0.02, 부호도 무작위 — 측정해서 확인함).
    a2j 매핑 자체는 위 joint_pos_target 비트일치 검사가 물리와 무관하게 이미 확정하므로,
    방향 검사는 부하 없는 조건으로 옮겨 실제로 검증 가능하게 한다."""
    # 이 테스트가 판별력을 가지려면 아티큘레이션 관절 순서가 정책 순서와 실제로 달라야 한다
    # (§9.3: [joint2, joint4, joint6, joint1, joint3, joint5] vs POLICY_JOINT_NAMES)
    assert tuple(env.robot.joint_names) != ol.POLICY_JOINT_NAMES
    base = torch.tensor(mdp.A_STAND, device=env.device).expand(env.num_envs, -1)
    airborne = env.scene.env_origins.clone()
    airborne[:, 2] += 1.0
    ident_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    zero_vel = torch.zeros(env.num_envs, 6, device=env.device)
    for k, name in enumerate(ol.POLICY_JOINT_NAMES):
        env.reset()
        env.robot.write_root_pose_to_sim(torch.cat([airborne, ident_quat], dim=1))
        env.robot.write_root_velocity_to_sim(zero_vel)
        act = base.clone()
        act[:, k] += 0.4
        for _ in range(3):                         # 0.3 s, 낙하 중(바닥 도달 전) 정착 확인
            env.step(act)
        j = env.robot.joint_names.index(name)
        tgt = env.robot.data.joint_pos_target
        torch.testing.assert_close(tgt[:, j], act[:, k], atol=1e-6, rtol=0)
        others = [i for i in range(6) if i != j]
        torch.testing.assert_close(
            tgt[:, others],
            base[:, [ol.POLICY_JOINT_NAMES.index(env.robot.joint_names[i]) for i in others]],
            atol=1e-6, rtol=0,
        )
        moved = env.robot.data.joint_pos[:, j] - env.robot.data.default_joint_pos[:, j]
        assert (moved > 0.15).all(), (name, moved)


def test_history_pairs_pre_step_quat_with_raw_action(env):
    """실기 순서 재현: index 0 = (a_t, a_t 작용 *전* 쿼터니언), 액션은 클립 전 raw (§3, §9.3).

    reset() 직후엔 root_quat_w 가 곧 리셋 자세라 "작용 전 q" 와 "리셋값 그대로" 를 구별할
    수 없다(리뷰 발견) — 한 스텝 먼저 밟아 리셋 자세에서 벗어난 뒤에 캡처·검증한다."""
    env.reset()
    env.step(_rand_actions(env))                       # 리셋 자세에서 벗어난다
    prev_act_his0 = env._act_his[:, 0].clone()          # 이번 스텝의 action_cost 가 쓸 prev_actions
    q_pre = ol.wxyz_to_xyzw(env.robot.data.root_quat_w).clone()
    act = torch.full((env.num_envs, ol.NUM_ACTIONS), 1.5, device=env.device)   # 한계(0.8727) 밖
    obs, _, terminated, _, extras = env.step(act)
    alive = ~env.reset_buf
    assert alive.any()
    o = obs["policy"][alive]
    torch.testing.assert_close(o[:, 0:4], q_pre[alive], atol=1e-6, rtol=0)     # 작용 전 q
    # index 1 = 직전 스텝의 작용 전 q = 리셋 자세(노이즈 포함). 둘이 달라야 이 검사가 판별력을 가진다
    assert (o[:, 0:4] - o[:, 4:8]).abs().max() > 1e-4
    assert torch.all(o[:, 16:22] == 1.5)                                       # raw, 클립 안 됨
    assert torch.all(o[:, 28:40] == ol.ACT_INIT)                               # index 2,3 은 아직 초기값
    # 관절 목표는 클립됐다
    assert env.robot.data.joint_pos_target.abs().max() <= ol.ACTION_LIMIT + 1e-6
    # action 보상항이 실제로 (raw_act, prev_act) 를 쓰는지 — _raw_act/_act 나 _act_his 인덱스가
    # 뒤바뀌어도 안 걸리던 구멍(리뷰 발견)을 이 값으로 메운다
    expected_action = ((act - prev_act_his0) ** 2).sum(-1).mean().item()
    assert float(extras["log"]["rew/action"]) == pytest.approx(expected_action, abs=1e-4)


def test_terminated_env_resets_with_init_history_and_death_reward(env):
    """env 3 을 뒤집어 놓으면 그 스텝에 terminated → 보상 = death(−1) 로 대체 → 리셋 →
    첫 관측은 리셋값 그대로. 다른 env 는 이력이 이어진다.

    브리프의 180° (x축) 대신 90° 만 건다 — 180° 는 다리가 바닥을 관통한 채로 물리가
    시작돼 접촉 폭발/NaN 위험이 있다(브리프 Step 2 의 완화안). 90° 도 TILT_THRESH(0.7,
    ≈82°) 를 기하로 이미 넘는다(||[0.7071,0,0,0.7071]-[0,0,0,1]|| ≈ 0.765) 이므로
    tilt 종료를 물리 없이도 보장한다."""
    env.reset()
    env.step(_rand_actions(env))
    tilt = torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0]], device=env.device)  # x 축 90° (w,x,y,z)
    pose = torch.cat([env.robot.data.root_pos_w[3:4], tilt], dim=1)
    env.robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([3], device=env.device))
    obs, rew, terminated, truncated, _ = env.step(_rand_actions(env))
    assert terminated[3] and not truncated[3]
    assert rew[3].item() == pytest.approx(env.cfg.reward_weights.death)
    o = obs["policy"]
    assert torch.all(o[3, :16].view(4, 4)[:, :3] == 0.0) and torch.all(o[3, :16].view(4, 4)[:, 3] == 1.0)
    assert torch.all(o[3, 16:] == ol.ACT_INIT)
    assert env.episode_length_buf[3] == 0
    survivors = (~env.reset_buf).nonzero().squeeze(-1)
    assert survivors.numel() > 0
    assert (o[survivors, 16:22] != ol.ACT_INIT).any()                          # 이력이 이어짐
