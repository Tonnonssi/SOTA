"""prepare_mjcf: floor/light 제거, 무명 바디 이름 부여, 나머지는 보존."""

import os
import xml.etree.ElementTree as ET

import pytest

from chair_rl import chair_asset
from chair_rl.mass_spec import BODY_NAMES


@pytest.fixture
def prepared(tmp_path):
    # 메시 상대경로 때문에 원본과 같은 디렉터리에 써야 하지만, 파싱만 하는 테스트는
    # 임시 디렉터리로 충분하다.
    dst = str(tmp_path / "chair_isaac.xml")
    out = chair_asset.prepare_mjcf(chair_asset.MJCF_SRC, dst)
    assert out == dst
    return ET.parse(dst).getroot()


def test_floor_and_lights_removed(prepared):
    wb = prepared.find("worldbody")
    assert wb.find("light") is None
    assert all(g.get("name") != "floor" for g in wb.iter("geom"))


def test_every_body_named_in_tree_order(prepared):
    names = [b.get("name") for b in prepared.find("worldbody").iter("body")]
    assert names == list(BODY_NAMES)


def test_joints_and_actuators_preserved(prepared):
    joints = sorted(filter(None, (j.get("name") for j in prepared.iter("joint"))))
    assert joints == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert prepared.find(".//freejoint").get("name") == "root"
    acts = sorted(a.get("joint") for a in prepared.find("actuator"))
    assert acts == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def test_source_untouched():
    # 규칙: mjcf/chair.xml 은 수정하지 않는다
    src = ET.parse(chair_asset.MJCF_SRC).getroot()
    assert src.find("worldbody").find("light") is not None
    unnamed = [b for b in src.find("worldbody").iter("body") if b.get("name") is None]
    assert len(unnamed) == 6
