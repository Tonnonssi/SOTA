# chair_asset + mass_spec 구현 계획 (이슈 #2, 뼈대 A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `mjcf/chair.xml`을 Isaac USD로 빌드하되 질량·관성·COM을 MuJoCo 값으로 구워 넣고, `chair_sim.py`가 그 USD를 쓰게 한다.

**Architecture:** `mass_spec.py`(순수 상수, Isaac 없음) → `chair_asset.py`(MJCF 전처리 → `MjcfConverter` → USD 후처리: 여분 아티큘레이션 루트 제거 + `MassAPI` authoring) → 스펙 해시별 캐시 디렉터리. 학습 env와 재생기가 같은 파일을 읽는다. 검증은 두 층: CPU 테스트(스펙 상수 = MuJoCo 계산값, MJCF 전처리)와 Isaac 테스트(빌드된 USD를 스폰해 `get_masses()`/`get_coms()` 되읽기).

**Tech Stack:** Isaac Lab 0.54.4 / Isaac Sim 5.1.0 (`~/miniforge3/envs/env_isaaclab/bin/python`, 3.11), pxr(USD), MuJoCo 3.8.1 (`~/miniforge3/envs/lerobot/bin/python` 에만 있음), pytest.

**Spec:** `docs/specs/2026-08-25-isaac-rl-design.md` — §1(배치), §2①(질량 오류), §2⑤(timestep), §9.2(에셋 빌드), §9.6(테스트). 이슈: https://github.com/Tonnonssi/SOTA/issues/2

## Global Constraints

- 브랜치는 이슈 하나에 대응한다. 이 계획의 브랜치는 `feat/2-chair-asset`. (CLAUDE.md)
- diff가 400줄을 넘으면 멈추고 분할안을 제시한다. (CLAUDE.md)
- 계획 밖의 문제를 발견하면 고치지 말고 멈추고 보고한다 — 라벨: 깊이초과 / 선행조건 / 전제무효화 / 인접유혹. (CLAUDE.md)
- PR 본문에 "이 코드가 틀렸다면 어떻게 틀렸을지"를 반드시 채운다. (CLAUDE.md)
- `mjcf/chair.xml`은 수정하지 않는다 — MuJoCo 호환 유지. 전처리 사본(`mjcf/chair_isaac.xml`)만 만든다. (§2, §9.2)
- 질량의 출처는 **MuJoCo 3.8.1이 `mjcf/chair.xml`을 읽은 값**이다. 다른 출처 금지. (§2①)
- 물리 timestep ≤ 0.008 s. `chair_sim.py`의 `1/120` 유지. (§2⑤)
- USD 단위는 metersPerUnit=1, kilogramsPerUnit=1 (실측). 단위 변환 없음.
- Isaac 모듈은 `AppLauncher` 기동 **뒤에만** import 가능. Kit은 프로세스당 한 번만 뜬다.
- 커밋 메시지는 레포 관례 `feat[isaac]: …` / `test[isaac]: …`, 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- 실행 파이썬: Isaac 관련은 `~/miniforge3/envs/env_isaaclab/bin/python`, MuJoCo 비교 테스트만 `~/miniforge3/envs/lerobot/bin/python`.

## 선행조건 (계획 시작 전 사람이 확인)

**`main`에 `isaac/`가 없다.** `chair_sim.py`·`keyframes.py`·`test_keyframes.py`는
`feat/isaac-keyframe-replay`(커밋 `ab8abcb`, 미병합, PR 없음)에만 있다. Task 5가 그
파일을 고치므로 **이 브랜치는 그것이 `main`에 병합된 뒤 `main`에서 딴다.** 병합 전에
시작해야 한다면 `feat/isaac-keyframe-replay`에서 따되, PR base도 그 브랜치로 잡는다.

---

## 파일 구조

| 파일 | 책임 | 만듦/고침 |
|---|---|---|
| `isaac/pyproject.toml` | `chair_rl` 패키지 + pytest 설정(`isaac` 마커) | 만듦 |
| `isaac/chair_rl/__init__.py` | 빈 파일 (gym 등록은 이슈 C) | 만듦 |
| `isaac/chair_rl/mass_spec.py` | `BodyMass`, `MassSpec`, `MUJOCO`, `with_servos()`, `spec_hash()` | 만듦 |
| `isaac/chair_rl/chair_asset.py` | `prepare_mjcf()`, `postprocess_usd()`, `build_usd()`, `articulation_cfg()` | 만듦 |
| `isaac/scripts/dump_mass_spec.py` | MuJoCo로 상수 블록 재생성 (MJCF가 바뀔 때) | 만듦 |
| `isaac/tests/conftest.py` | `sys.path`에 `isaac/` 추가, `--isaac` 옵션, Kit 세션 픽스처 | 만듦 |
| `isaac/tests/test_mass_spec.py` | 상수 = MuJoCo 값 (mujoco 있을 때만) | 만듦 |
| `isaac/tests/test_prepare_mjcf.py` | 전처리: floor/light 제거, 바디 이름, 관절 보존 | 만듦 |
| `isaac/tests/test_asset_build.py` | Isaac: USD authoring 값, 루트 1개, 스폰 후 질량·COM 되읽기 | 만듦 |
| `isaac/chair_sim.py` | `chair_asset` 사용, `--inspect`에 질량 출력, `--servo-mass` | 고침 |
| `.gitignore` | `isaac/usd/` 추가 (빌드 산출물) | 고침 |

`chair_asset.py`의 Isaac import는 **함수 안**에 둔다. `prepare_mjcf()`는 순수 XML 처리라
Kit 없이 테스트된다.

---

### Task 1: 브랜치와 패키지 골격

**Files:**
- Create: `isaac/pyproject.toml`
- Create: `isaac/chair_rl/__init__.py`
- Create: `isaac/tests/__init__.py` (빈 파일)
- Create: `isaac/tests/conftest.py`
- Modify: `.gitignore` (끝에 추가)

**Interfaces:**
- Produces: `chair_rl` 패키지가 `import chair_rl`로 풀린다 (pip -e 또는 conftest의 sys.path). pytest 마커 `isaac`, 옵션 `--isaac`, 픽스처 `kit_app`.

- [ ] **Step 1: 브랜치 생성**

```bash
git checkout main && git pull
git checkout -b feat/2-chair-asset
```

- [ ] **Step 2: pyproject.toml 작성**

`isaac/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "chair_rl"
version = "0.0.1"
description = "Chair-type tripedal robot: Isaac Lab RL env (paper arXiv:2404.05932)"
requires-python = ">=3.11"
dependencies = ["numpy"]

[tool.setuptools]
packages = ["chair_rl"]

[tool.pytest.ini_options]
testpaths = ["."]
markers = [
    "isaac: Isaac Sim(Kit) 기동이 필요한 테스트. `--isaac` 없이는 skip.",
]
```

- [ ] **Step 3: 패키지·테스트 디렉터리와 conftest**

```bash
mkdir -p isaac/chair_rl isaac/tests isaac/scripts
: > isaac/chair_rl/__init__.py
: > isaac/tests/__init__.py
```

`isaac/tests/conftest.py`:

```python
"""isaac/ 테스트 공통 설정.

- `chair_rl` 을 pip 설치 없이도 import 할 수 있게 isaac/ 를 sys.path 에 넣는다
  (MuJoCo 비교 테스트는 chair_rl 이 설치되지 않은 lerobot 파이썬에서 돈다).
- Kit 이 필요한 테스트는 `@pytest.mark.isaac` 을 달고, `--isaac` 옵션이 있을 때만 돈다.
  Kit 은 프로세스당 한 번만 뜨므로 세션 픽스처다.
"""

import os
import sys

import pytest

ISAAC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ISAAC_DIR not in sys.path:
    sys.path.insert(0, ISAAC_DIR)


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", default=False,
                     help="Isaac Sim(Kit) 을 띄우는 테스트를 실행한다")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        return
    skip = pytest.mark.skip(reason="--isaac 옵션 없음")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def kit_app(request):
    """헤드리스 Kit. 세션 끝에 닫는다."""
    if not request.config.getoption("--isaac"):
        pytest.skip("--isaac 옵션 없음")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True})
    app = launcher.app
    yield app
    app.close()
```

- [ ] **Step 4: .gitignore에 빌드 산출물 추가**

`.gitignore` 끝에:

```
# Isaac: MJCF -> USD 빌드 산출물 (chair_asset.build_usd 가 만든다)
isaac/usd/
mjcf/chair_isaac.xml
```

`mjcf/chair_isaac.xml`이 이미 추적 중이면 인덱스에서만 뺀다:

```bash
git ls-files --error-unmatch mjcf/chair_isaac.xml 2>/dev/null && git rm --cached mjcf/chair_isaac.xml
```

- [ ] **Step 5: 설치·수집 확인**

```bash
~/miniforge3/envs/env_isaaclab/bin/python -m pip install -e isaac/
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest --collect-only -q | tail -3
```

Expected: 에러 없이 `test_keyframes.py`의 기존 테스트 13개가 수집된다.

- [ ] **Step 6: Commit**

```bash
git add isaac/pyproject.toml isaac/chair_rl/__init__.py isaac/tests/__init__.py isaac/tests/conftest.py .gitignore
git commit -m "chore[isaac]: chair_rl 패키지 골격과 pytest isaac 마커

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: mass_spec.py — MuJoCo 질량 상수

**Files:**
- Create: `isaac/chair_rl/mass_spec.py`
- Create: `isaac/scripts/dump_mass_spec.py`
- Test: `isaac/tests/test_mass_spec.py`

**Interfaces:**
- Produces:
  - `BodyMass(mass: float, com: tuple[float,float,float], inertia: tuple[float,float,float], axes_wxyz: tuple[float,float,float,float])` — frozen dataclass. `com`은 바디 프레임 m, `inertia`는 주축 관성 kg·m², `axes_wxyz`는 주축 방향 쿼터니언 (w,x,y,z).
  - `MassSpec(bodies: dict[str, BodyMass], servo_mass: bool = False)` — frozen dataclass. `total_mass() -> float`, `with_servos() -> MassSpec`, `spec_hash() -> str` (8 hex).
  - `MUJOCO: MassSpec` — 바디 8개 (`dummy, chair, bracket1..3, leg1..3`).
  - `ROOT_BODY = "dummy"`, `BODY_NAMES: tuple[str, ...]` (MJCF 트리 순서).

- [ ] **Step 1: 실패하는 테스트 작성**

`isaac/tests/test_mass_spec.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_mass_spec.py -v
```

Expected: FAIL — `ImportError: cannot import name 'mass_spec'` (또는 `AttributeError`).

- [ ] **Step 3: 구현**

`isaac/chair_rl/mass_spec.py`:

```python
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
```

`isaac/scripts/dump_mass_spec.py`:

```python
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
```

- [ ] **Step 4: 통과 확인 (두 파이썬 모두)**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_mass_spec.py -v
cd isaac && ~/miniforge3/envs/lerobot/bin/python -m pytest tests/test_mass_spec.py -v
```

Expected: 첫 번째는 4 passed + `test_matches_mujoco` skipped. 두 번째는 5 passed.

- [ ] **Step 5: Commit**

```bash
git add isaac/chair_rl/mass_spec.py isaac/scripts/dump_mass_spec.py isaac/tests/test_mass_spec.py
git commit -m "feat[isaac]: mass_spec — MuJoCo 가 계산한 바디 질량·관성·COM 상수

Isaac MJCF 임포터가 density 를 무시하고 볼록껍질 x 1000 kg/m^3 을 쓰므로
(138 g -> 1725 g) 질량의 출처를 MuJoCo 로 못 박는다. dummy 는 무질량이라
PhysX 용 미소질량을 준다. 서보 54 g 은 옵션(점질량, COM·관성 불변).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: prepare_mjcf — 전처리 (순수 XML, Kit 불필요)

**Files:**
- Create: `isaac/chair_rl/chair_asset.py` (이 태스크에서는 `prepare_mjcf`만)
- Test: `isaac/tests/test_prepare_mjcf.py`

**Interfaces:**
- Produces: `prepare_mjcf(src: str, dst: str) -> str` — `src`를 읽어 `<worldbody>`의 `<light>`와 `name="floor"` geom을 제거하고, `name` 없는 `<body>`에 첫 `<geom>`의 이름을 붙여 `dst`에 쓴다. `dst` 경로를 반환. `dst`는 `src`와 같은 디렉터리여야 한다(메시 상대경로 `./mesh/…`).
- 상수: `REPO`, `MJCF_SRC = REPO/mjcf/chair.xml`, `MJCF_DST = REPO/mjcf/chair_isaac.xml`, `USD_ROOT = REPO/isaac/usd`.

- [ ] **Step 1: 실패하는 테스트 작성**

`isaac/tests/test_prepare_mjcf.py`:

```python
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
    joints = sorted(j.get("name") for j in prepared.iter("joint"))
    assert joints == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert prepared.find("worldbody//freejoint").get("name") == "root"
    acts = sorted(a.get("joint") for a in prepared.find("actuator"))
    assert acts == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def test_source_untouched():
    # 규칙: mjcf/chair.xml 은 수정하지 않는다
    src = ET.parse(chair_asset.MJCF_SRC).getroot()
    assert src.find("worldbody").find("light") is not None
    unnamed = [b for b in src.find("worldbody").iter("body") if b.get("name") is None]
    assert len(unnamed) == 6
```

- [ ] **Step 2: 실패 확인**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_prepare_mjcf.py -v
```

Expected: FAIL — `ImportError: cannot import name 'chair_asset'`.

- [ ] **Step 3: 구현**

`isaac/chair_rl/chair_asset.py`:

```python
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
```

- [ ] **Step 4: 통과 확인**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_prepare_mjcf.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add isaac/chair_rl/chair_asset.py isaac/tests/test_prepare_mjcf.py
git commit -m "feat[isaac]: chair_asset.prepare_mjcf — floor/light 제거, 무명 바디 이름 부여

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: build_usd / postprocess_usd / articulation_cfg + Isaac 빌드 테스트

**Files:**
- Modify: `isaac/chair_rl/chair_asset.py` (함수 3개 추가)
- Test: `isaac/tests/test_asset_build.py`

**Interfaces:**
- Consumes: `prepare_mjcf()` (Task 3), `MassSpec`/`MUJOCO`/`BODY_NAMES`/`ROOT_BODY` (Task 2).
- Produces:
  - `build_usd(spec: MassSpec = MUJOCO, force: bool = False) -> str` — `USD_ROOT/<spec.spec_hash()>/chair.usd`. 캐시: 디렉터리에 `.postprocessed` 파일이 있고 내용이 해시와 같으면 그대로 반환. `force=True`면 재변환·재후처리.
  - `postprocess_usd(usd_path: str, spec: MassSpec) -> None` — 파일을 열어 `worldBody`의 `ArticulationRootAPI`/`PhysxArticulationAPI`를 떼고, 바디 프림 `<default>/dummy/<name>`에 `MassAPI` 속성 4개를 authoring한 뒤 루트 레이어를 저장.
  - `articulation_cfg(usd_path: str, prim_path: str = "/World/Robot", spawn_height: float = 0.12, joint_pos: dict[str, float] | None = None, effort_limit: float = 0.3) -> ArticulationCfg` — `chair_sim.py`의 `build_robot_cfg`와 같은 액추에이터 파라미터(kp 40, damping 0.01, armature 0.001).
  - 바디 프림 경로 규칙: 변환된 USD의 defaultPrim 아래 `dummy/<body>`. 실측: `/chair_isaac/dummy/{dummy,chair,_body_0..5}` — 전처리 후에는 `_body_N`이 `bracket1`… 으로 바뀐다.

- [ ] **Step 1: 실패하는 테스트 작성**

`isaac/tests/test_asset_build.py`:

```python
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


def test_cache_hit_returns_same_path(kit_app, usd_path):
    from chair_rl import chair_asset

    assert chair_asset.build_usd(MUJOCO) == usd_path
    assert chair_asset.build_usd(MUJOCO.with_servos(), force=True) != usd_path


def test_physics_layer_masses_and_coms(kit_app, usd_path):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from chair_rl import chair_asset

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
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
```

- [ ] **Step 2: 실패 확인**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_asset_build.py --isaac -v 2>&1 | grep -E "PASS|FAIL|Error|passed|failed"
```

Expected: FAIL — `AttributeError: module 'chair_rl.chair_asset' has no attribute 'build_usd'`. (Kit 기동에 ~40 s.)

- [ ] **Step 3: 구현 — chair_asset.py 에 추가**

`prepare_mjcf` 아래에 이어서:

```python
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
```

- [ ] **Step 4: 통과 확인**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_asset_build.py --isaac -v 2>&1 | grep -E "PASS|FAIL|Error|passed|failed|assert"
```

Expected: 3 passed.

**`test_physics_layer_masses_and_coms`의 COM 검사가 실패하면** — 질량은 맞는데 COM만 어긋나면 임포터가 바디 프레임을 보존하지 않는 것이다. 이것은 설계문서 열린 질문 #2의 **부정적 판정**이고 계획 밖이다. **고치지 말고 멈추고 보고한다 — 라벨: 전제무효화.** 어긋난 바디 이름과 (스펙 com, 되읽은 com) 값을 함께 적는다.

- [ ] **Step 5: CPU 테스트가 여전히 도는지**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest -q 2>&1 | tail -2
```

Expected: `test_keyframes` 13 + `test_mass_spec` 4 + `test_prepare_mjcf` 4 passed, isaac 3 skipped.

- [ ] **Step 6: Commit**

```bash
git add isaac/chair_rl/chair_asset.py isaac/tests/test_asset_build.py
git commit -m "feat[isaac]: chair_asset.build_usd — MJCF->USD 빌드에 MuJoCo 질량을 굽는다

임포터 결과를 후처리해 저장한다: worldBody 의 여분 아티큘레이션 루트 제거,
바디 8개에 MassAPI(mass/centerOfMass/diagonalInertia/principalAxes) authoring.
스펙 해시별 디렉터리에 캐시하고, 학습 env 와 재생기가 같은 파일을 읽는다.
Isaac 테스트가 스폰 후 get_masses/get_coms 되읽기로 프레임 보존을 판정한다.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: chair_sim.py 를 chair_asset 위로

**Files:**
- Modify: `isaac/chair_sim.py` — 함수 `prepare_mjcf`, `convert_mjcf`, `build_robot_cfg`, `_strip_extra_articulation_roots` 삭제; `main()`의 해당 호출 교체; `--inspect`에 질량 출력; `--servo-mass` 플래그 추가.

**Interfaces:**
- Consumes: `chair_asset.build_usd(spec, force)`, `chair_asset.articulation_cfg(usd_path, spawn_height, joint_pos)`, `mass_spec.MUJOCO`.
- 외부 동작 불변: 플래그 `--motion/--joint-order/--policy-hz/--script-*/--spawn-height/--force-convert/--inspect/--no-ground/--warmup/--init-pose` 전부 유지. 바뀌는 것은 재생되는 모델의 질량(1725 g → 138 g)뿐이다.

- [ ] **Step 1: 기준 동작 기록 (변경 전)**

```bash
~/miniforge3/envs/env_isaaclab/bin/python -u isaac/chair_sim.py --inspect 2>&1 | grep "^\[chair\]"
```

기록해 둔다. 관절 6개, 바디 8개(`_body_0..5`)가 찍혀야 한다.

- [ ] **Step 2: import 교체**

`from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg` 줄을 지우고, `import torch` 아래에:

```python
from chair_rl import chair_asset  # noqa: E402
from chair_rl.mass_spec import MUJOCO  # noqa: E402
```

argparse에 (`--force-convert` 다음):

```python
parser.add_argument(
    "--servo-mass",
    action="store_true",
    help="MJCF 에 빠진 SG90 6개(54 g)를 점질량으로 더한 모델을 쓴다 (설계문서 §2②)",
)
```

- [ ] **Step 3: 함수 4개 삭제, main() 교체**

`prepare_mjcf`, `convert_mjcf`, `build_robot_cfg`, `_strip_extra_articulation_roots` 정의를 통째로 지운다. `main()` 첫 부분을:

```python
def main() -> None:
    spec = MUJOCO.with_servos() if args.servo_mass else MUJOCO
    usd_path = chair_asset.build_usd(spec, force=args.force_convert)
    info(f"USD: {usd_path}  (mass spec {spec.spec_hash()}, servo_mass={spec.servo_mass})")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    sim.set_camera_view(eye=[0.6, -0.6, 0.4], target=[0.1, 0.1, 0.08])

    if not args.no_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.92))
    light.func("/World/light", light)

    robot = Articulation(
        chair_asset.articulation_cfg(
            usd_path, spawn_height=args.spawn_height, joint_pos=_initial_joint_pos()
        )
    )
    sim.reset()

    info(f"articulation joints ({robot.num_joints}): {robot.joint_names}")
    info(f"bodies ({robot.num_bodies}): {robot.body_names}")
    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    info("masses (g): " + ", ".join(f"{n}={m*1000:.2f}" for n, m in zip(robot.body_names, masses))
         + f"  | total={masses.sum()*1000:.2f}")

    if args.inspect:
        simulation_app.close()
        return
```

이하(`order = JOINT_ORDERS[...]`부터)는 그대로 둔다. `_strip_extra_articulation_roots(...)` 호출 줄은 지운다.

- [ ] **Step 4: 동작 확인**

```bash
~/miniforge3/envs/env_isaaclab/bin/python -u isaac/chair_sim.py --inspect 2>&1 | grep "^\[chair\]"
```

Expected: 바디 이름이 `['dummy', 'chair', 'bracket1', 'leg1', 'bracket2', 'leg2', 'bracket3', 'leg3']`(순서는 임포터가 정한다), `total=138.03`. 그리고:

```bash
~/miniforge3/envs/env_isaaclab/bin/python -u isaac/chair_sim.py --inspect --servo-mass 2>&1 | grep -E "total=|servo_mass"
```

Expected: `servo_mass=True`, `total=192.03`.

```bash
timeout 300 ~/miniforge3/envs/env_isaaclab/bin/python -u isaac/chair_sim.py --motion script --headless --script-steps 2 2>&1 | grep "^\[chair\]" | tail -5
```

Expected: 예외 없이 `script 재생 완료`까지 진행. (`base z`는 기존 로그와 달라질 수 있다 — 질량이 바뀌었으므로 회귀가 아니다.)

- [ ] **Step 5: 기존 keyframes 테스트 + 전체**

```bash
cd isaac && ~/miniforge3/envs/env_isaaclab/bin/python -m pytest -q 2>&1 | tail -2
```

Expected: 21 passed, 3 skipped.

- [ ] **Step 6: diff 규모 확인**

```bash
git diff --stat main | tail -1
```

400줄을 넘으면 **멈추고 분할안을 제시한다** (예: Task 5를 별도 이슈로).

- [ ] **Step 7: Commit**

```bash
git add isaac/chair_sim.py
git commit -m "feat[isaac]: chair_sim 이 chair_asset 의 USD 를 쓴다 — 재생 모델 질량 1725 g -> 138 g

prepare_mjcf/convert_mjcf/build_robot_cfg/_strip_extra_articulation_roots 를
chair_asset 로 옮겼다. 외부 플래그는 그대로, --servo-mass 추가, --inspect 에
바디별 질량 출력.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: PR

- [ ] **Step 1: 푸시**

```bash
git push -u origin feat/2-chair-asset
```

- [ ] **Step 2: PR 생성**

```bash
gh pr create --base main --title "feat[isaac]: chair_asset + mass_spec — USD 빌드에 MuJoCo 질량을 굽는다 (#2)" --body "$(cat <<'EOF'
Closes #2. 설계문서 §2①, §9.2.

## 무엇
- `chair_rl.mass_spec`: MuJoCo 3.8.1 이 `mjcf/chair.xml` 에서 계산한 바디 8개의 질량·주축관성·COM. 서보 54 g 옵션.
- `chair_rl.chair_asset`: MJCF 전처리(무명 바디 이름 부여) → USD 변환 → 후처리(여분 아티큘레이션 루트 제거, MassAPI authoring) → 스펙 해시별 캐시.
- `chair_sim.py` 가 이 USD 를 쓴다. 재생 모델 질량 1725 g → 138.03 g.

## 검증
- CPU: `test_mass_spec`(MuJoCo 비교, lerobot 파이썬), `test_prepare_mjcf`
- Isaac: `test_asset_build --isaac` — 파일 층(authoring 값, 루트 1개) + 물리 층(`get_masses` 합 138.03 g, 바디별 `get_coms` = 스펙 COM)
- `chair_sim.py --inspect` 합계 138.03 g / `--servo-mass` 192.03 g

## 이 코드가 틀렸다면 어떻게 틀렸을지
1. **COM 프레임.** `get_coms()` 되읽기가 스펙과 일치하는 것으로 "임포터가 바디 프레임을 보존한다"고 판정했다. 그러나 PhysX 가 authored COM 을 그대로 되돌려주는 것일 뿐 실제 관성 계산에 다른 프레임을 쓴다면 테스트는 통과하고 물리만 틀린다. 반증 방법: 서 있는 로봇의 정적 평형에서 다리 반력 분포를 MuJoCo 와 비교.
2. **principalAxes 쿼터니언 규약.** MuJoCo `iquat` 는 (w,x,y,z), USD `Gf.Quatf(real, imag)` 로 넣었다. 규약이 어긋나면 질량·COM 테스트는 통과하고 관성 방향만 틀린다 — 대칭에 가까운 leg 에서는 안 보이고 chair(비대칭 좌면+등받이)에서만 드러난다.
3. **dummy 미소질량 0.1 g.** PhysX 가 질량비가 극단적인 링크(0.1 g 루트 ↔ 123 g 자식)에서 관절 솔버가 불안정해질 수 있다. 증상: 서 있을 때 관절 떨림. 그러면 dummy 질량을 올리고 chair 에서 그만큼 빼는 것이 대안이다.
4. **서보 점질량 근사.** `with_servos()` 는 질량만 더하고 COM·관성을 안 바꾼다. 서보는 브래킷 끝에 있으므로 실제로는 관성이 커진다. 이 옵션으로 §2 게이트가 통과해도 "서보 질량 때문"이라고 결론 내리면 안 된다 — 근사가 틀린 방향으로 도운 것일 수 있다.

## 범위 밖
- RL env (이슈 B/C), 액추에이터 튜닝, `mjcf/chair.xml` 수정.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage (§9.2):** 전처리·이름 부여(Task 3) ✓, 변환+후처리·루트 제거·MassAPI(Task 4) ✓, 해시 캐시(Task 4 `build_usd`) ✓, 서보 옵션(Task 2 `with_servos`) ✓, `chair_sim.py` 이식(Task 5) ✓, 빌드 테스트 `get_coms` 되읽기(Task 4) ✓, §2⑤ timestep(`articulation_cfg`는 timestep을 안 정하고 `chair_sim`/env의 `1/120` 유지) ✓. §9.6의 `test_mass_spec`·`test_asset_build` ✓. `--inspect` 질량 출력(이슈 #2 검증 방법) ✓.

**Type consistency:** `MassSpec.spec_hash()` (Task 2) ↔ `build_usd`의 `spec.spec_hash()` (Task 4) ↔ `chair_sim`의 `spec.spec_hash()` (Task 5) 일치. `articulation_cfg(usd_path, prim_path, spawn_height, joint_pos, effort_limit)` 시그니처가 Task 4 정의와 Task 5 호출(`spawn_height=`, `joint_pos=`)에서 일치. `BODY_NAMES` 튜플이 Task 2·3·4에서 같은 8개.

**알려진 불확실성 (계획이 아니라 실행 시 판정):** Task 4 Step 4의 COM 검사. 실패 시 절차가 적혀 있다(멈추고 보고, 전제무효화).
