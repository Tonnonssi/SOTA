"""mass_spec 상수가 MuJoCo 가 mjcf/chair.xml 에서 계산한 값과 같은지.

mujoco 는 lerobot 파이썬에만 있다:
    ~/miniforge3/envs/lerobot/bin/python -m pytest isaac/tests/test_mass_spec.py
없으면 MuJoCo 비교는 skip 되고 순수 로직 테스트만 돈다.
"""

import hashlib
import os

import numpy as np
import pytest

from chair_rl import mass_spec as ms

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJCF = os.path.join(REPO, "mjcf", "chair.xml")


def test_body_names_and_root():
    assert ms.ROOT_BODY == "dummy"
    assert ms.BODY_NAMES == ("dummy", "chair", "bracket1", "leg1",
                             "bracket2", "leg2", "bracket3", "leg3")
    assert set(ms.MUJOCO.bodies) == set(ms.BODY_NAMES)


def test_total_mass_is_138g():
    # MuJoCo 3.8.1 실측 138.0297650 g. dummy 는 0 이 아니라 PhysX 용 미소질량이다.
    assert ms.MUJOCO.total_mass() == pytest.approx(0.138030, abs=2e-4)
    assert ms.MUJOCO.bodies["dummy"].mass < 1e-3


def test_with_servos_adds_54g_only_to_seat_and_brackets():
    s = ms.MUJOCO.with_servos()
    assert s.servo_mass is True
    assert s.total_mass() - ms.MUJOCO.total_mass() == pytest.approx(0.054, abs=1e-9)
    for b in ("bracket1", "bracket2", "bracket3"):
        assert s.bodies[b].mass - ms.MUJOCO.bodies[b].mass == pytest.approx(0.009)
        assert s.bodies[b].com == ms.MUJOCO.bodies[b].com          # 점질량 근사: COM 불변
        assert s.bodies[b].inertia == ms.MUJOCO.bodies[b].inertia  # 관성 불변
    assert s.bodies["chair"].mass - ms.MUJOCO.bodies["chair"].mass == pytest.approx(0.027)
    for b in ("leg1", "leg2", "leg3", "dummy"):
        assert s.bodies[b] == ms.MUJOCO.bodies[b]


def test_spec_hash_is_stable_and_distinguishes_servos():
    h = ms.MUJOCO.spec_hash()
    assert len(h) == 8 and all(c in "0123456789abcdef" for c in h)
    assert ms.MUJOCO.spec_hash() == h                       # 결정적
    assert ms.MUJOCO.with_servos().spec_hash() != h         # 스펙이 다르면 해시가 다르다


def test_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(MJCF)
    # MJCF 의 무명 바디는 mj_id2name 이 None 을 준다. 트리 순서가 BODY_NAMES 와 같다.
    for i, name in enumerate(ms.BODY_NAMES, start=1):
        spec = ms.MUJOCO.bodies[name]
        if name == "dummy":
            assert m.body_mass[i] == 0.0     # MuJoCo 에서 dummy 는 무질량
            continue
        assert spec.mass == pytest.approx(m.body_mass[i], rel=1e-6)
        np.testing.assert_allclose(spec.com, m.body_ipos[i], rtol=0, atol=1e-8)
        np.testing.assert_allclose(spec.inertia, m.body_inertia[i], rtol=1e-6)
        # 쿼터니언은 부호 반전이 같은 회전이다
        q_spec, q_mj = np.array(spec.axes_wxyz), m.body_iquat[i]
        assert min(np.abs(q_spec - q_mj).max(), np.abs(q_spec + q_mj).max()) < 1e-6
