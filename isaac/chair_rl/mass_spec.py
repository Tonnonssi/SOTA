"""바디별 질량·관성·COM 상수. 출처: MuJoCo 3.8.1 이 mjcf/chair.xml 을 읽은 값.

Isaac 의 MJCF 임포터는 density="175.5" 를 무시하고 1000 kg/m^3 x 볼록껍질 부피를
쓴다(설계문서 §2①: 138 g 모델이 1725 g 로 들어온다). 그래서 USD 를 믿지 않고 이
상수를 MassAPI 로 구워 넣는다(§9.2).

값을 다시 뽑으려면 (MJCF 가 바뀌었을 때):
    ~/miniforge3/envs/lerobot/bin/python isaac/scripts/dump_mass_spec.py

프레임: com 은 바디 프레임 m. MJCF 에서 chair/bracket/leg 의 바디 프레임은 전부
CAD 원점과 일치하고, 변환된 USD 프림의 로컬 변환도 항등이라(2026-08-26 실측)
MuJoCo body_ipos/body_iquat 를 그대로 USD centerOfMass/principalAxes 에 넣으면 된다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

ROOT_BODY = "dummy"
# MJCF 트리 순서. Isaac 이 임포트한 _body_0..5 와의 대응은 조인트 연결로 확인했다:
# joint2: chair->bracket1, joint1: bracket1->leg1, joint4: chair->bracket2, ...
BODY_NAMES = ("dummy", "chair", "bracket1", "leg1", "bracket2", "leg2", "bracket3", "leg3")

# PhysX 는 질량 0 인 강체를 받지 않는다. dummy 는 geom 이 없어 MuJoCo 에서 0 이므로
# 무시 가능한 값을 준다. (0.1 g, 138 g 의 0.07%)
DUMMY_MASS = 1.0e-4
DUMMY_INERTIA = 1.0e-9

# SG90 서보 9 g x 6. MJCF 주석 기준 servo2/3/6 은 좌면, servo1/4/5 는 각 브래킷.
SERVO_G = 0.009
SERVO_ADD = {"chair": 3 * SERVO_G, "bracket1": SERVO_G, "bracket2": SERVO_G, "bracket3": SERVO_G}


@dataclass(frozen=True)
class BodyMass:
    mass: float                                   # kg
    com: tuple[float, float, float]               # m, 바디 프레임
    inertia: tuple[float, float, float]           # kg m^2, 주축
    axes_wxyz: tuple[float, float, float, float]  # 주축 방향 (w, x, y, z)


@dataclass(frozen=True)
class MassSpec:
    bodies: dict[str, BodyMass]
    servo_mass: bool = False

    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies.values())

    def with_servos(self) -> "MassSpec":
        """빠진 서보 54 g 을 점질량으로 더한다 — 질량만, COM·관성 불변 (§2② 알려진 한계)."""
        bodies = dict(self.bodies)
        for name, add in SERVO_ADD.items():
            bodies[name] = replace(bodies[name], mass=bodies[name].mass + add)
        return MassSpec(bodies=bodies, servo_mass=True)

    def spec_hash(self) -> str:
        """스펙 내용의 해시 8 hex. USD 캐시 디렉터리 이름에 쓴다."""
        payload = {
            name: [round(b.mass, 12), [round(v, 12) for v in b.com],
                   [round(v, 15) for v in b.inertia], [round(v, 12) for v in b.axes_wxyz]]
            for name, b in sorted(self.bodies.items())
        }
        payload["servo_mass"] = self.servo_mass
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


# --- MuJoCo 3.8.1, mjcf/chair.xml, 2026-08-26 (scripts/dump_mass_spec.py 출력) ---
MUJOCO = MassSpec(bodies={
    "dummy": BodyMass(
        mass=DUMMY_MASS,
        com=(0.0, 0.0, 0.0),
        inertia=(DUMMY_INERTIA, DUMMY_INERTIA, DUMMY_INERTIA),
        axes_wxyz=(1.0, 0.0, 0.0, 0.0),
    ),
    "chair": BodyMass(
        mass=1.228703176e-01,
        com=(6.255555393e-02, 8.052600840e-02, 1.322640236e-01),
        inertia=(7.162255949e-04, 5.568592030e-04, 3.476070136e-04),
        axes_wxyz=(-2.686514053e-02, 8.908307263e-01, 1.422726888e-02, 4.533171806e-01),
    ),
    "bracket1": BodyMass(
        mass=1.249719506e-03,
        com=(1.241731913e-01, 4.921163323e-02, 1.026752068e-01),
        inertia=(2.502203471e-07, 2.412192532e-07, 4.491498183e-08),
        axes_wxyz=(5.833583482e-01, 7.161578021e-01, 2.002063422e-01, -3.266932211e-01),
    ),
    "leg1": BodyMass(
        mass=3.803429647e-03,
        com=(1.366917581e-01, 3.645258485e-02, 5.743079690e-02),
        inertia=(3.057212407e-06, 3.049653940e-06, 1.360181280e-07),
        axes_wxyz=(9.997288734e-01, 2.039808629e-04, 2.328385946e-02, -4.750749799e-06),
    ),
    "bracket2": BodyMass(
        mass=1.249719296e-03,
        com=(6.040942355e-02, 4.865763777e-02, 1.026752065e-01),
        inertia=(2.502203088e-07, 2.412192164e-07, 4.491497993e-08),
        axes_wxyz=(3.648327870e-01, 6.435036305e-01, 1.814896025e-01, 6.479673135e-01),
    ),
    "leg2": BodyMass(
        mass=3.803429639e-03,
        com=(4.715039217e-02, 3.614343557e-02, 5.743079709e-02),
        inertia=(3.057212474e-06, 3.049654011e-06, 1.360181199e-07),
        axes_wxyz=(7.069184251e-01, 1.631993417e-02, -1.660841071e-02, 7.069117064e-01),
    ),
    "bracket3": BodyMass(
        mass=1.249720189e-03,
        com=(6.097422792e-02, 1.133770939e-01, 1.026751995e-01),
        inertia=(2.502205290e-07, 2.412194429e-07, 4.491495012e-08),
        axes_wxyz=(6.511261514e-01, 1.870984054e-01, 6.418955471e-01, 3.591643470e-01),
    ),
    "leg3": BodyMass(
        mass=3.803429125e-03,
        com=(4.843532827e-02, 1.261164298e-01, 5.743079182e-02),
        inertia=(3.057211771e-06, 3.049653322e-06, 1.360181035e-07),
        axes_wxyz=(7.069184253e-01, -1.631992645e-02, 1.660840675e-02, 7.069117065e-01),
    ),
})
