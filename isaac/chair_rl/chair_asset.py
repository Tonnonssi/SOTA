"""Chair-type tripedal robot 에셋: mjcf/chair.xml -> Isaac USD.

설계문서 §9.2. 파이프라인:
    mjcf/chair.xml
      -> prepare_mjcf():   floor/light 제거 + 무명 <body> 에 이름 부여
      -> MjcfConverter:    USD 생성 (isaac/usd/<spec-hash>/chair.usd)
      -> postprocess_usd(): 여분 ArticulationRootAPI 제거 + MassAPI 로 질량·관성·COM 굽기
학습 env 와 재생기(chair_sim.py)가 같은 파일을 읽는다.

Isaac import 는 함수 안에 둔다 — prepare_mjcf() 는 Kit 없이 돌고 테스트된다.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .mass_spec import MUJOCO, MassSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJCF_SRC = os.path.join(REPO, "mjcf", "chair.xml")
MJCF_DST = os.path.join(REPO, "mjcf", "chair_isaac.xml")   # 메시 상대경로 때문에 같은 디렉터리
USD_ROOT = os.path.join(REPO, "isaac", "usd")
USD_FILE = "chair.usd"
POSTPROCESS_MARK = ".postprocessed"


def prepare_mjcf(src: str = MJCF_SRC, dst: str = MJCF_DST) -> str:
    """로봇만 남긴 MJCF 사본을 만들어 경로를 돌려준다.

    원본 <worldbody> 의 바닥 평면과 조명은 임포트하면 (1) worldBody 가 별도 아티큘레이션
    루트가 되고 (2) 바닥이 로봇 USD 안에 들어가 스폰 변환을 같이 받는다. 둘 다 Isaac 쪽에서
    따로 만들므로 제거한다.

    무명 <body> 는 임포터가 _body_N 으로 이름 짓고 순서 보장이 없다. 첫 <geom> 의 이름
    (bracket1, leg1, ...) 을 붙여 질량 스펙·관절 매핑이 이름으로 돌게 한다.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    for child in list(worldbody):
        if child.tag == "light" or (child.tag == "geom" and child.get("name") == "floor"):
            worldbody.remove(child)
    for body in worldbody.iter("body"):
        if body.get("name") is None:
            geom = body.find("geom")
            if geom is None or geom.get("name") is None:
                raise ValueError(f"이름 없는 body 에 이름 있는 geom 이 없다: {ET.tostring(body)[:80]}")
            body.set("name", geom.get("name"))
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    return dst


def _usd_dir(spec: MassSpec) -> str:
    return os.path.join(USD_ROOT, spec.spec_hash())


def build_usd(spec: MassSpec = MUJOCO, force: bool = False) -> str:
    """MJCF -> USD 빌드(캐시) 후 경로를 돌려준다. Kit 기동 이후에만 부를 수 있다."""
    usd_dir = _usd_dir(spec)
    usd_path = os.path.join(usd_dir, USD_FILE)
    mark = os.path.join(usd_dir, POSTPROCESS_MARK)
    if not force and os.path.isfile(usd_path) and os.path.isfile(mark):
        with open(mark) as f:
            if f.read().strip() == spec.spec_hash():
                return usd_path

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from isaacsim.core.utils.extensions import enable_extension

    # MJCF 임포터는 isaaclab kit 앱에 기본 활성화돼 있지 않다.
    enable_extension("isaacsim.asset.importer.mjcf")

    mjcf_path = prepare_mjcf()
    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=usd_dir,
        usd_file_name=USD_FILE,
        fix_base=False,          # MJCF freejoint = 떠 있는 베이스
        make_instanceable=False,
        import_sites=True,
        self_collision=False,
        force_usd_conversion=True,   # 후처리로 파일을 고치므로 컨버터 캐시는 쓰지 않는다
    )
    converter = MjcfConverter(cfg)
    postprocess_usd(converter.usd_path, spec)
    with open(mark, "w") as f:
        f.write(spec.spec_hash())
    return converter.usd_path


def postprocess_usd(usd_path: str, spec: MassSpec) -> None:
    """임포터 결과를 고쳐 저장한다: 여분 아티큘레이션 루트 제거 + MassAPI authoring."""
    from pxr import Gf, PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    base = stage.GetDefaultPrim().GetPath()
    keep = f"{base}/dummy/dummy"

    # ① MJCF 임포터가 worldBody 에도 아티큘레이션 루트를 붙인다. 두 개면 Isaac Lab 이
    #    '/World/Robot' 아래에서 아티큘레이션을 특정하지 못하고 RuntimeError 를 낸다.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(base)):
        if str(prim.GetPath()) != keep and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    # ② 질량·관성·COM. 임포터는 MassAPI 를 적용만 하고 값을 authoring 하지 않아
    #    PhysX 가 충돌 지오메트리 x 기본밀도 1000 으로 계산한다(§2①). 여기서 덮어쓴다.
    for name, body in spec.bodies.items():
        prim = stage.GetPrimAtPath(f"{base}/dummy/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"바디 프림 없음: {base}/dummy/{name} — prepare_mjcf 의 이름 부여를 확인")
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr().Set(float(body.mass))
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*body.com))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*body.inertia))
        w, x, y, z = body.axes_wxyz
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        api.CreateDensityAttr().Set(0.0)   # 질량이 authored 되면 밀도는 무시되지만 명시한다

    stage.GetRootLayer().Save()


def articulation_cfg(
    usd_path: str,
    prim_path: str = "/World/Robot",
    spawn_height: float = 0.12,
    joint_pos: dict[str, float] | None = None,
    effort_limit: float = 0.3,
):
    """chair_sim.py 의 build_robot_cfg 와 같은 액추에이터 파라미터.

    MJCF actuator: position kp=40 / joint damping .010, armature .001 / forcerange +-0.3.
    effort_limit 기본값은 MJCF 의 0.3 을 유지한다. 실측 SG90 스톨토크 0.1764 로 낮추는
    결정은 §2 system ID 에서 한다.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_height),
            joint_pos=joint_pos if joint_pos is not None else {"joint[1-6]": 0.0},
        ),
        actuators={
            "servos": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=40.0,
                damping=0.01,
                armature=0.001,
                effort_limit_sim=effort_limit,
            )
        },
    )
