"""Chair-Walk-Direct-v0 — 걷기 DirectRLEnv (설계문서 §9.3, §9.5).

env 는 얇은 껍데기다: 상태(이력 버퍼·potentials·작용 전 쿼터니언)만 들고, 수식은 전부
obs_layout / mdp 의 순수 함수에 있다. 훅 순서는 DirectRLEnv.step() 소스 그대로:
  _pre_physics_step → ×12 _apply_action → _get_dones → _get_rewards → _reset_idx → _get_observations

실기 rl_walk.py 와 맞춘 세 가지 (§3, §9.3):
  1. 이력 index 0 = (a_t, a_t 작용 *전* 쿼터니언). 실기는 publish 직후 IMU 를 읽는다.
     그래서 _pre_physics_step 이 root_quat_w 를 clone 해 두고 _get_observations 가 그걸 push 한다.
  2. 이력의 액션은 클립 전 raw, 관절 목표만 ±ACTION_LIMIT 클립.
  3. 리셋된 env 의 첫 관측은 리셋값 그대로 (push 건너뜀). reset() 경로에서도 성립하도록
     reset_buf 가 아니라 _skip_push 마스크를 쓴다.

위치는 env 로컬(p = root_pos_w − env_origins). p_target=(10,0,0) 은 env 원점 기준이다.
base_env 추출은 stand_env 이슈에서 (이슈 #12 결정).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from . import chair_asset, mdp
from . import obs_layout as ol
from .mass_spec import MUJOCO


@configclass
class WalkEnvCfg(DirectRLEnvCfg):
    # 시간: 1/120 × 12 = 0.1 s = mdp.CONTROL_DT, 35 s / 0.1 = 350 = mdp.MAX_EPISODE_LEN (§9.5, §2⑤)
    decimation = 12
    episode_length_s = mdp.MAX_EPISODE_LEN * mdp.CONTROL_DT
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    observation_space = ol.OBS_DIM
    action_space = ol.NUM_ACTIONS
    state_space = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.6, replicate_physics=True)

    # 에셋 (§9.2): 질량 스펙은 MuJoCo 그대로가 기준선, 서보 54 g 은 §2 system ID 의 실험 변수
    servo_mass = False
    effort_limit = 0.3

    # 초기 상태 (§5 walk): yaw 는 랜덤화하지 않는다 — p_target/heading 이 +x 고정
    init_height = 0.101          # 서 있을 때 dummy z 실측 (§3)
    init_height_noise = 0.002
    init_joint_noise = 0.02
    init_tilt_noise = 0.02       # roll, pitch 각각

    # 보상 가중치 (Table IV) — 종료 임계값은 mdp 상수 (논문값, 튜닝 대상 아님)
    reward_weights: mdp.WalkRewardWeights = mdp.WalkRewardWeights()


class WalkEnv(DirectRLEnv):
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 시간 상수의 단일 출처는 mdp 다 (이슈 #6). cfg 가 어긋나면 progress 의 dt 와 truncation 이 틀어진다.
        assert abs(self.step_dt - mdp.CONTROL_DT) < 1e-9, self.step_dt
        assert self.max_episode_length == mdp.MAX_EPISODE_LEN, self.max_episode_length

        # 정책 인덱스 → 관절 인덱스. 아티큘레이션 순서는 [joint2, joint4, joint6, joint1, joint3, joint5]
        # (임포터의 폭우선 순회, §9.3) 이라 정책 순서와 다르다. 이 한 곳에서만 만든다.
        self._a2j, names = self.robot.find_joints(list(ol.POLICY_JOINT_NAMES), preserve_order=True)
        assert tuple(names) == ol.POLICY_JOINT_NAMES, names

        n, dev = self.num_envs, self.device
        self._a_stand = torch.tensor(mdp.A_STAND, device=dev)
        self._p_target = torch.tensor(mdp.P_TARGET, device=dev)
        self._rot_his, self._act_his = ol.new_history(n, dev)
        self._raw_act = torch.full((n, ol.NUM_ACTIONS), ol.ACT_INIT, device=dev)
        self._act = self._raw_act.clamp(-ol.ACTION_LIMIT, ol.ACTION_LIMIT)
        self._quat_pre = torch.zeros(n, 4, device=dev)
        self._quat_pre[:, 0] = 1.0                       # (w,x,y,z) 단위
        self._potentials = torch.zeros(n, device=dev)
        self._skip_push = torch.zeros(n, dtype=torch.bool, device=dev)
        self._term_reasons: dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------- scene

    def _setup_scene(self):
        spec = MUJOCO.with_servos() if self.cfg.servo_mass else MUJOCO
        usd_path = chair_asset.build_usd(spec)
        joint_pos = {name: float(mdp.A_STAND[i]) for i, name in enumerate(ol.POLICY_JOINT_NAMES)}
        self.robot = Articulation(chair_asset.articulation_cfg(
            usd_path, prim_path="/World/envs/env_.*/Robot",
            spawn_height=self.cfg.init_height, joint_pos=joint_pos,
            effort_limit=self.cfg.effort_limit,
        ))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.92))
        light.func("/World/light", light)

    # ---------------------------------------------------------------- helpers

    def root_pos_local(self) -> torch.Tensor:
        """좌면 중심(dummy) 위치, env 원점 기준 (N,3)."""
        return self.robot.data.root_pos_w - self.scene.env_origins

    def _root_quat_xyzw(self) -> torch.Tensor:
        return ol.wxyz_to_xyzw(self.robot.data.root_quat_w)

    # ---------------------------------------------------------------- hooks (순서 = step())

    def _pre_physics_step(self, actions: torch.Tensor):
        self._raw_act = actions.clone()                                   # 이력용 (클립 전)
        self._act = actions.clamp(-ol.ACTION_LIMIT, ol.ACTION_LIMIT)      # 관절용
        self._quat_pre = self.robot.data.root_quat_w.clone()              # clone 필수 — 뷰면 물리 뒤 덮인다

    def _apply_action(self):
        self.robot.set_joint_position_target(self._act, joint_ids=self._a2j)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, self._term_reasons = mdp.walk_terminated(self.root_pos_local(), self._root_quat_xyzw())
        truncated = mdp.walk_truncated(self.episode_length_buf)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # prev_actions = 이력 index 0 = 직전 스텝의 raw 액션 (push 는 _get_observations 에서, 이 뒤에)
        terms, self._potentials = mdp.walk_reward_terms(
            self.root_pos_local(), self._root_quat_xyzw(), self._potentials,
            self._raw_act, self._act_his[:, 0], self.robot.data.joint_vel,
            dt=mdp.CONTROL_DT, p_target=self._p_target,
        )
        reward = mdp.walk_total(terms, self.cfg.reward_weights, self.reset_terminated)
        log = {f"rew/{k}": v.mean() for k, v in terms.items()}
        log.update({f"term/{k}": v.float().mean() for k, v in self._term_reasons.items()})
        self.extras["log"] = log
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        k, dev, c = len(env_ids), self.device, self.cfg

        # 관절: a_stand ± noise, 정책 순서 → 관절 순서
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._a2j] = self._a_stand + sample_uniform(
            -c.init_joint_noise, c.init_joint_noise, (k, ol.NUM_ACTIONS), dev)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), None, env_ids)

        # 루트: env 원점 + (0, 0, init_height ± noise), roll/pitch ± noise, yaw 0, 속도 0
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] = self.scene.env_origins[env_ids]
        root[:, 2] += c.init_height + sample_uniform(-c.init_height_noise, c.init_height_noise, k, dev)
        roll = sample_uniform(-c.init_tilt_noise, c.init_tilt_noise, k, dev)
        pitch = sample_uniform(-c.init_tilt_noise, c.init_tilt_noise, k, dev)
        root[:, 3:7] = quat_from_euler_xyz(roll, pitch, torch.zeros(k, device=dev))
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        # progress 의 기준점. root_pos_w 는 다음 update 전까지 갱신되지 않으므로 방금 쓴 값으로 (env 로컬)
        self._potentials[env_ids] = mdp.potential(root[:, :3] - self.scene.env_origins[env_ids], self._p_target)

        # 이력 리셋은 반드시 여기 — 관측은 리셋 *뒤에* 계산된다 (§9.3)
        ol.reset_history(self._rot_his, self._act_his, env_ids)
        self._skip_push[env_ids] = True

    def _get_observations(self) -> dict:
        obs = ol.push(self._rot_his, self._act_his,
                      ol.wxyz_to_xyzw(self._quat_pre), self._raw_act,
                      skip_mask=self._skip_push)
        self._skip_push[:] = False
        return {"policy": obs}
