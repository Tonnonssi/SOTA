"""isaac/ 테스트 공통 설정.

- `chair_rl` 을 pip 설치 없이도 import 할 수 있게 isaac/ 를 sys.path 에 넣는다
  (MuJoCo 비교 테스트는 chair_rl 이 설치되지 않은 lerobot 파이썬에서 돈다).
- Kit 이 필요한 테스트는 `@pytest.mark.isaac` 을 달고, `--isaac` 옵션이 있을 때만 돈다.
  Kit 은 프로세스당 한 번만 뜨므로 세션 픽스처다.
"""

import os
import sys

import pytest

ISAAC_DIR = os.path.dirname(os.path.abspath(__file__))
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
        if item.get_closest_marker("isaac"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def kit_app(request):
    """헤드리스 Kit. 세션 끝에 닫는다.

    fast_shutdown(기본 True) 이면 close() 가 C++ 에서 exit 0 으로 프로세스를 끝내 pytest 가
    실패를 보고하지 못한다(요약/junitxml 없음). 그래서 끈다.
    graceful close 는 SimulationContext 의 timeline STOP 핸들러(`while not playing: render()`)
    를 건드려 무한 루프에 빠질 수 있으므로, 남은 컨텍스트를 먼저 정리한다.
    """
    if not request.config.getoption("--isaac"):
        pytest.skip("--isaac 옵션 없음")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "fast_shutdown": False})
    app = launcher.app
    yield app
    from isaaclab.sim import SimulationContext

    if SimulationContext.instance() is not None:
        SimulationContext.instance().clear_all_callbacks()
        SimulationContext.clear_instance()
    app.close()
