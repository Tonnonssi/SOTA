#!/usr/bin/env python3
"""mjcf/chair.xml 을 MuJoCo 로 읽어 mass_spec.py 의 MUJOCO 블록을 다시 찍는다.

    ~/miniforge3/envs/lerobot/bin/python isaac/scripts/dump_mass_spec.py

출력을 mass_spec.py 의 해당 블록에 붙여넣고 tests/test_mass_spec.py 를 돌린다.
"""

import os

import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAMES = ("dummy", "chair", "bracket1", "leg1", "bracket2", "leg2", "bracket3", "leg3")


def fmt(v):
    return "(" + ", ".join(f"{x:.9e}" for x in v) + ")"


def main():
    m = mujoco.MjModel.from_xml_path(os.path.join(REPO, "mjcf", "chair.xml"))
    print(f"# --- MuJoCo {mujoco.__version__}, mjcf/chair.xml ---")
    for i, name in enumerate(NAMES, start=1):
        if name == "dummy":
            continue  # 무질량. mass_spec.py 의 DUMMY_* 상수를 쓴다
        print(f'    "{name}": BodyMass(')
        print(f"        mass={m.body_mass[i]:.9e},")
        print(f"        com={fmt(m.body_ipos[i])},")
        print(f"        inertia={fmt(m.body_inertia[i])},")
        print(f"        axes_wxyz={fmt(m.body_iquat[i])},")
        print("    ),")
    print(f"# total = {m.body_mass.sum():.9e} kg")


if __name__ == "__main__":
    main()
