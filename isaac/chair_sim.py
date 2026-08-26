#!/usr/bin/env python3
"""Chair-Type Asymmetrical Tripedal Robot -> Isaac Sim / Isaac Lab.

mjcf/chair.xml 을 USD로 변환해 스폰하고, 레포에 포함된 학습된 ONNX 정책
(models/walk.onnx, models/stand.onnx)으로 6개 서보 관절을 구동한다.

  python -u isaac/chair_sim.py --inspect                    # 관절 순서만 확인
  python -u isaac/chair_sim.py --motion script --headless   # 고정 키프레임 재생
  python -u isaac/chair_sim.py --motion sine --headless     # 정책 없이 구동 확인
  python -u isaac/chair_sim.py --motion walk --livestream 1 # 원격 송출
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--motion",
    default="walk",
    choices=["walk", "stand", "script", "sine", "hold"],
    help="walk/stand = 레포의 ONNX 정책, script = config.py의 고정 키프레임 재생 "
    "(걷기->일어서기->걷기), sine = 사인파 스윕, hold = 정지",
)
parser.add_argument(
    "--joint-order",
    default="tree",
    choices=["tree", "actuator", "tree-reversed"],
    help="정책 출력 6개를 어느 관절에 매핑할지. 기본 tree = MJCF 트리 순회 순서",
)
parser.add_argument("--policy-hz", type=float, default=10.0, help="정책 추론 주기")
parser.add_argument(
    "--script-hz",
    type=float,
    default=20.0,
    help="--motion script 의 키프레임 재생 주기. 실기 connect_performing.py 가 "
    "rospy.Rate(20) 으로 뿌리므로 기본값도 20",
)
parser.add_argument(
    "--script-steps", type=int, default=3, help="--motion script 의 앞/뒤 걸음 수"
)
parser.add_argument(
    "--script-loop",
    action="store_true",
    help="--motion script 를 큐 끝에서 처음으로 되감아 계속 반복 재생",
)
parser.add_argument("--spawn-height", type=float, default=0.12, help="초기 스폰 높이 (m)")
parser.add_argument("--force-convert", action="store_true", help="USD 캐시 무시하고 재변환")
parser.add_argument("--inspect", action="store_true", help="관절 이름/순서를 출력하고 종료")
parser.add_argument("--no-ground", action="store_true", help="바닥 평면을 추가하지 않음")
parser.add_argument(
    "--warmup",
    type=float,
    default=2.0,
    help="정책을 넘기기 전에 STANDING_POS를 유지할 시간(초). 원본 rl_walk.py가 "
    "SLEEPING->STANDING 으로 자세를 잡고 시작하는 것을 흉내낸다",
)
parser.add_argument(
    "--init-pose",
    default="standing",
    choices=["standing", "zero"],
    help="초기 관절 자세",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.inspect:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Isaac 모듈은 앱 기동 이후에만 import 가능하다 ---------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Usd  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

# MJCF의 관절 정의 순서(= 바디 트리 순회 순서). Isaac Gym이 이 MJCF를 읽을 때의
# DOF 순서와 같으므로, Isaac Gym에서 학습된 정책의 출력 순서로 이것을 기본값으로 둔다.
#
# 주의: 레포만으로는 확정할 수 없다. embedded.ino는 서보 인덱스를 핀 번호로만
# 매핑하고 관절 이름을 남기지 않으며, src/rl_walk.py의 simRad2realDeg()는 sim 인덱스를
# 뒤집어 실기 서보에 보낸다(real[i] = sim[5-i]). 실기 서보 번호와 joint 이름의 대응이
# 문서화돼 있지 않아 아래 세 후보 중 어느 것인지 단정할 수 없다.
# 걸음새가 이상하면 --joint-order 로 바꿔가며 확인할 것.
JOINT_ORDERS = {
    "tree": ["joint2", "joint1", "joint4", "joint3", "joint6", "joint5"],
    "actuator": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    "tree-reversed": ["joint5", "joint6", "joint3", "joint4", "joint1", "joint2"],
}

JOINT_LIMIT = 0.872665  # MJCF joint range, rad (= 50도)

# config.py의 STANDING_POS(실기 서보 각도, deg) = [90, 80, 90, 100, 90, 100] 를
# sim 좌표로 되돌린 값. src/rl_walk.py 의 simRad2realDeg() 역변환:
#   real_deg[i] = -rad2deg(sim_rad[5-i]) + 90   =>   sim_rad[j] = deg2rad(90 - real_deg[5-j])
# 정책 출력과 같은 인덱스 순서(0..5)를 따른다.
STANDING_SIM = [-0.17453, 0.0, -0.17453, 0.0, 0.17453, 0.0]


def info(msg: str) -> None:
    """Kit이 stdout을 가로채서 print()가 사라지므로 stderr로 내보낸다."""
    print(f"[chair] {msg}", file=sys.stderr, flush=True)


class OnnxPolicy:
    """레포의 ONNX 정책. obs = [회전 이력 4x4, 액션 이력 4x6] = 40차원.

    src/rl_walk.py 의 구성을 그대로 따른다. 다만 실기 IMU 보정(quat의 x,y 부호 반전)은
    적용하지 않는다 - 그건 하드웨어 IMU 장착 방향 보정이고, 여기서는 시뮬레이터가
    학습 때와 같은 좌표계의 몸통 자세를 직접 준다.
    """

    def __init__(self, model_path: str, num_rot_his: int = 4, num_act_his: int = 4):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        raw_shape = self.session.get_inputs()[0].shape
        # 동적 축(문자열)은 1로 고정한다
        self.input_shape = [d if isinstance(d, int) else 1 for d in raw_shape]

        self.rot_his = np.zeros([num_rot_his, 4], dtype=np.float32)
        self.rot_his[:, 3] = 1.0  # 단위 쿼터니언 (x,y,z,w)
        self.act_his = np.ones([num_act_his, 6], dtype=np.float32)

    def step(self, quat_xyzw: np.ndarray) -> np.ndarray:
        obs = np.concatenate(
            [self.rot_his.flatten(), self.act_his.flatten()], 0
        ).astype(np.float32)
        action = self.session.run(
            ["mu"], {self.input_name: obs.reshape(self.input_shape)}
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(1, 6)

        # 원본과 동일하게 액션을 낸 뒤에 이력을 갱신한다 (최신이 앞)
        self.rot_his = np.concatenate([quat_xyzw.reshape(1, 4), self.rot_his], 0)[:-1]
        self.act_his = np.concatenate([action, self.act_his], 0)[:-1]
        return action[0]


def prepare_mjcf() -> str:
    """로봇만 남긴 MJCF를 만들어 그 경로를 돌려준다.

    원본 chair.xml의 <worldbody>에는 바닥 평면(geom name="floor")과 조명이 들어 있다.
    그대로 임포트하면 두 가지 문제가 생긴다:
      1. worldBody가 별도의 아티큘레이션 루트가 되어 Isaac Lab이 '/World/Robot' 아래에서
         아티큘레이션을 하나로 특정하지 못하고 RuntimeError를 낸다.
      2. 바닥이 로봇 USD 안에 들어가 있어서 로봇 스폰 변환(z=spawn_height)을 같이 받아
         공중에 뜬 바닥이 된다.
    바닥과 조명은 Isaac 쪽에서 따로 만들므로 여기서 제거한다.

    메쉬 경로가 './mesh/...' 상대 경로라 원본과 같은 디렉터리에 써야 해석된다.
    """
    import xml.etree.ElementTree as ET

    src = os.path.join(REPO, "mjcf", "chair.xml")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    dst = os.path.join(REPO, "mjcf", "chair_isaac.xml")

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    removed = []
    for child in list(worldbody):
        if child.tag == "light":
            worldbody.remove(child)
            removed.append("light")
        elif child.tag == "geom" and child.get("name") == "floor":
            worldbody.remove(child)
            removed.append("geom:floor")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    info(f"MJCF 전처리: {os.path.basename(dst)} (제거: {removed})")
    return dst


def convert_mjcf() -> str:
    """MJCF -> USD. 이미 변환돼 있으면 캐시를 쓴다."""
    mjcf_path = prepare_mjcf()

    # MJCF 임포터는 isaaclab의 kit 앱에 기본 활성화돼 있지 않다. 켜지 않으면
    # MjcfConverter가 "Can't execute command: MJCFCreateImportConfig" 로 죽는다.
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.mjcf")

    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.join(REPO, "isaac", "usd"),
        usd_file_name="chair_isaac.usd",
        fix_base=False,  # MJCF의 freejoint = 떠 있는 베이스
        make_instanceable=False,
        import_sites=True,
        self_collision=False,
        force_usd_conversion=args.force_convert,
    )
    converter = MjcfConverter(cfg)
    info(f"USD: {converter.usd_path}")
    return converter.usd_path


def _initial_joint_pos() -> dict[str, float]:
    """초기 관절 각도를 {관절이름: rad} 로 만든다."""
    if args.init_pose == "zero":
        return {"joint[1-6]": 0.0}
    order = JOINT_ORDERS[args.joint_order]
    return {name: STANDING_SIM[i] for i, name in enumerate(order)}


def build_robot_cfg(usd_path: str) -> ArticulationCfg:
    # MJCF actuator: position, kp=40, forcerange +-0.3 / joint damping .010, armature .001
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args.spawn_height),
            joint_pos=_initial_joint_pos(),
        ),
        actuators={
            "servos": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=40.0,
                damping=0.01,
                armature=0.001,
                effort_limit_sim=0.3,
            )
        },
    )


def _strip_extra_articulation_roots(root_path: str, keep: str) -> None:
    """`keep` 이외의 프림에서 ArticulationRootAPI를 떼어낸다.

    MJCF 임포터가 worldBody에도 아티큘레이션 루트를 붙이는 경우가 있어, MJCF 전처리로
    막지 못한 경우를 대비한 안전장치다.
    """
    import isaacsim.core.utils.stage as stage_utils
    from pxr import PhysxSchema, UsdPhysics

    stage = stage_utils.get_current_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        path = str(prim.GetPath())
        if path == keep:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            info(f"여분 아티큘레이션 루트 제거: {path}")


def main() -> None:
    usd_path = convert_mjcf()

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    sim.set_camera_view(eye=[0.6, -0.6, 0.4], target=[0.1, 0.1, 0.08])

    if not args.no_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.92))
    light.func("/World/light", light)

    robot = Articulation(build_robot_cfg(usd_path))
    _strip_extra_articulation_roots("/World/Robot", keep="/World/Robot/dummy/dummy")
    sim.reset()

    info(f"articulation joints ({robot.num_joints}): {robot.joint_names}")
    info(f"bodies ({robot.num_bodies}): {robot.body_names}")

    if args.inspect:
        simulation_app.close()
        return

    order = JOINT_ORDERS[args.joint_order]
    missing = [j for j in order if j not in robot.joint_names]
    if missing:
        raise RuntimeError(
            f"관절 {missing} 를 아티큘레이션에서 찾을 수 없음. 실제: {robot.joint_names}"
        )
    # 정책 출력 인덱스 -> 아티큘레이션 관절 인덱스
    action_to_joint = [robot.joint_names.index(j) for j in order]
    info(f"joint-order={args.joint_order} -> {order}")

    policy = None
    if args.motion in ("walk", "stand"):
        model_path = os.path.join(REPO, "models", f"{args.motion}.onnx")
        policy = OnnxPolicy(model_path)
        info(f"policy: {model_path} (input {policy.input_shape})")

    player = None
    if args.motion == "script":
        import keyframes

        queue = keyframes.build_walk_rise_walk(args.script_steps)
        player = keyframes.KeyframePlayer(queue, loop=args.script_loop)
        info(
            f"script: {len(queue)} keyframes @{args.script_hz}Hz "
            f"(~{len(queue) / args.script_hz:.1f}s), steps={args.script_steps}, "
            f"loop={args.script_loop}"
        )

    dt = sim.get_physics_dt()
    if args.motion == "script":
        # 스크립트는 정책보다 빠른 주기로 뿌린다(실기 20Hz). --policy-hz 는 무시된다.
        decimation = max(1, int(round((1.0 / args.script_hz) / dt)))
        info(f"dt={dt:.5f}s, keyframe every {decimation} steps (~{args.script_hz}Hz)")
    else:
        decimation = max(1, int(round((1.0 / args.policy_hz) / dt)))
        info(f"dt={dt:.5f}s, policy every {decimation} steps (~{args.policy_hz}Hz)")
    info(f"준비 완료. 스트리밍 클라이언트를 지금 붙여도 된다.")

    default_targets = robot.data.default_joint_pos.clone()
    targets = default_targets.clone()
    step = 0

    while simulation_app.is_running():
        if step % decimation == 0:
            if step * dt < args.warmup:
                # 자세 잡는 구간: 정책을 돌리지 않고 STANDING_POS를 유지한다
                targets = default_targets.clone()
            elif player is not None:
                row = player.next()
                targets = default_targets.clone()
                for a_idx, j_idx in enumerate(action_to_joint):
                    targets[0, j_idx] = float(row[a_idx])
                if player.just_finished():
                    if player.loop:
                        # 되감을 때 로봇은 넘어진 채로 남아 있다. 다음 바퀴는
                        # 서 있는 자세를 전제로 하므로 몸통을 초기 상태로 되돌린다.
                        robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                        robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                        info(f"script {player.laps}회 완료 (t={step * dt:.1f}s). 되감아 반복.")
                    else:
                        info(f"script 재생 완료 (t={step * dt:.1f}s). 마지막 자세 유지.")
            elif policy is not None:
                # Isaac Lab의 root_quat_w 는 (w,x,y,z) -> 정책은 (x,y,z,w)
                q = robot.data.root_quat_w[0].cpu().numpy()
                quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
                action = policy.step(quat_xyzw)
                action = np.clip(action, -JOINT_LIMIT, JOINT_LIMIT)
                targets = default_targets.clone()
                for a_idx, j_idx in enumerate(action_to_joint):
                    targets[0, j_idx] = float(action[a_idx])
            elif args.motion == "sine":
                t = step * dt
                targets = default_targets.clone()
                for a_idx, j_idx in enumerate(action_to_joint):
                    phase = 2.0 * np.pi * (0.5 * t + a_idx / 6.0)
                    targets[0, j_idx] = 0.6 * JOINT_LIMIT * np.sin(phase)
            # motion == "hold" 이면 targets 그대로 둔다

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if step % int(2.0 / dt) == 0:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            h = float(pos[2])
            q = robot.data.joint_pos[0].cpu().numpy()
            info(
                f"t={step * dt:6.1f}s  base=({pos[0]:+.3f}, {pos[1]:+.3f}, {h:+.3f})m  "
                f"q=[{', '.join(f'{v:+.2f}' for v in q)}]"
            )
            if not np.isfinite(h):
                raise RuntimeError("시뮬레이션이 발산했다 (base_z=NaN)")
        step += 1


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # simulation_app.close()는 fastShutdown 경로에서 os._exit(0)을 호출해
        # 종료 코드 0에 트레이스백까지 삼킨다. 닫기 전에 반드시 찍어둔다.
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
