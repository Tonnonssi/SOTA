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
