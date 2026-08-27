"""빌드된 USD 검증. Kit 필요:
    cd isaac && python -m pytest tests/test_asset_build.py --isaac -v

두 층으로 본다.
  1. 파일 층: MassAPI 가 authored 됐는가, 아티큘레이션 루트가 정확히 하나인가.
  2. 물리 층: 스폰해서 PhysX 가 되돌려주는 질량·COM 이 스펙과 같은가 —
     "임포터가 MJCF 바디 프레임을 보존한다" 는 전제(설계문서 열린 질문 #2)의 판정.
"""

import numpy as np
import pytest

from chair_rl.mass_spec import BODY_NAMES, MUJOCO, ROOT_BODY

pytestmark = pytest.mark.isaac


@pytest.fixture(scope="module")
def usd_path(kit_app):
    from chair_rl import chair_asset

    return chair_asset.build_usd(MUJOCO, force=True)


def test_file_layer_mass_authored_and_single_root(kit_app, usd_path):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert [str(p.GetPath()) for p in roots] == [f"{stage.GetDefaultPrim().GetPath()}/dummy/dummy"]

    base = stage.GetDefaultPrim().GetPath()
    for name in BODY_NAMES:
        prim = stage.GetPrimAtPath(f"{base}/dummy/{name}")
        assert prim.IsValid(), f"바디 프림 없음: {name}"
        m = UsdPhysics.MassAPI(prim)
        spec = MUJOCO.bodies[name]
        assert m.GetMassAttr().Get() == pytest.approx(spec.mass, rel=1e-6)
        np.testing.assert_allclose(list(m.GetCenterOfMassAttr().Get()), spec.com, atol=1e-7)
        np.testing.assert_allclose(list(m.GetDiagonalInertiaAttr().Get()), spec.inertia, rtol=1e-5)
        q = m.GetPrincipalAxesAttr().Get()          # Gf.Quatf: real + imaginary
        got = np.array([q.GetReal(), *q.GetImaginary()])
        want = np.array(spec.axes_wxyz)
        # q 와 -q 는 같은 회전이다
        assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-6, name

    # 프레임 보존: MuJoCo com 을 무변환으로 넣는 전제는 "바디 프림의 로컬 변환이 항등" 이다.
    # get_coms() 되읽기는 authored 값의 왕복이라 이것을 증명하지 못한다 — 여기서 직접 본다.
    from pxr import Gf, UsdGeom

    for name in BODY_NAMES:
        prim = stage.GetPrimAtPath(f"{base}/dummy/{name}")
        xf = UsdGeom.Xformable(prim).GetLocalTransformation()
        if name == ROOT_BODY:
            # dummy 만 MJCF 의 freejoint 위치(좌면 중심)에 놓인다: mjcf/chair.xml <body name="dummy" pos=…>
            np.testing.assert_allclose(list(xf.ExtractTranslation()), (0.095, 0.0785, 0.10365), atol=1e-6)
        else:
            assert Gf.IsClose(xf, Gf.Matrix4d(1.0), 1e-6), f"{name} 의 로컬 변환이 항등이 아님: {xf}"


def test_cache_hit_returns_same_path(kit_app, usd_path):
    from chair_rl import chair_asset

    assert chair_asset.build_usd(MUJOCO) == usd_path
    assert chair_asset.build_usd(MUJOCO.with_servos(), force=True) != usd_path


def test_physics_layer_masses_and_coms(kit_app, usd_path):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from chair_rl import chair_asset

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
    try:
        robot = Articulation(chair_asset.articulation_cfg(usd_path))
        sim.reset()

        assert sorted(robot.joint_names) == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        assert set(robot.body_names) == set(BODY_NAMES)
        assert robot.body_names[0] == ROOT_BODY

        masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
        assert masses.sum() == pytest.approx(MUJOCO.total_mass(), abs=1e-5)
        for i, name in enumerate(robot.body_names):
            assert masses[i] == pytest.approx(MUJOCO.bodies[name].mass, rel=1e-4), name

        # COM 은 바디(프림) 프레임. 스펙 com 과 같으면 프레임이 보존된 것이다.
        coms = robot.root_physx_view.get_coms()[0].cpu().numpy()   # (bodies, 7): xyz + quat
        for i, name in enumerate(robot.body_names):
            if name == ROOT_BODY:
                continue
            np.testing.assert_allclose(coms[i, :3], MUJOCO.bodies[name].com, atol=1e-4, err_msg=name)
    finally:
        # clear_instance() 는 콜백만 지우고 타임라인은 멈추지 않는다 — playing 인 채로 남으면 다음에
        # 만들어지는 SimulationContext 가 reset()/step() 에서 "재생될 때까지" 도는 무한 루프에 빠진다
        # (test_env_smoke.py 와 이 파일을 같은 세션에서 돌리면 재현됨 — 진단 근거는 task-2-report.md).
        # 그래서 stop() 도 반드시 불러야 한다 — 다만 순서가 중요하다: stop() 은 (터미널 실행이 아닐 때)
        # 안에서 app.update() 를 불러 STOP 이벤트를 동기 디스패치하는데, _app_control_on_stop_handle
        # 콜백이 아직 구독돼 있으면 그 콜백이 "재생될 때까지" render() 를 도는 무한 루프에 빠진다(대화형
        # 스크립트가 창을 열어 두게 하려는 의도된 동작 — reset() 은 이걸 피하려고 _disable_app_control_on_stop_handle 을
        # 잠깐 켠다). clear_instance() 가 그 구독을 해제하므로, stop() 은 반드시 clear_instance() *뒤에* 부른다.
        sim.clear_all_callbacks()
        sim_utils.SimulationContext.clear_instance()   # classmethod — conftest 의 teardown 과 같은 형태
        sim.stop()
