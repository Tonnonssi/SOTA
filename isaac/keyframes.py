#!/usr/bin/env python3
"""실기 고정 키프레임(src/config.py)을 Isaac sim 관절 각도 큐로 바꾼다.

논문/레포의 걷기·일어서기 동작은 학습된 정책이 아니라 손으로 잡은 자세들
(STANDING/WALKING_POS1~4/EXTENTION/ROLLED/SLEEPING)을 선형보간해 20Hz로 서보에
흘려보내는 방식이다(src/connect_performing.py). 여기서는 그 큐를 그대로 만들고,
서보 각도(deg)를 sim 관절 각도(rad)로 되돌려 Isaac 쪽에서 재생할 수 있게 한다.

Isaac 의존이 없으므로 numpy 만으로 임포트/테스트할 수 있다.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# src/config.py 는 numpy 만 의존한다(ROS 불필요). 값을 복사하지 않고 그대로 쓴다.
import config as cfg  # noqa: E402


def safe_clip(deg: np.ndarray) -> np.ndarray:
    """src/*.py 의 safeClip() 과 동일: [ANGLE_MIN, ANGLE_MAX] 클립 후 정수 변환.

    실기가 서보에 보내는 값이 정확히 이것이다. ROLLED_POS 에 150 이 들어 있어
    이 클립을 빼먹으면 MJCF 관절 한계(±50도)를 넘는 목표각이 나온다.
    """
    return np.clip(np.atleast_2d(np.asarray(deg, dtype=float)),
                   cfg.ANGLE_MIN, cfg.ANGLE_MAX).astype(int)


def real_deg_to_sim_rad(real_deg: np.ndarray) -> np.ndarray:
    """실기 서보 각도(deg) -> sim 관절 각도(rad). (N,6) 반환.

    src/rl_walk.py 의 simRad2realDeg() 역변환:
        real_deg[i] = -rad2deg(sim_rad[5-i]) + 90
        =>  sim_rad[j] = deg2rad(90 - real_deg[5-j])
    인덱스 뒤집기가 들어가므로 열 순서를 반대로 뒤집는다.
    """
    a = np.atleast_2d(np.asarray(real_deg, dtype=float))
    return np.deg2rad(90.0 - a[:, ::-1])


def linspace(start: np.ndarray, end: np.ndarray, step: int) -> np.ndarray:
    """src/*.py 의 linspace() 와 동일한 보간. 양 끝점을 포함한 (step,6)."""
    L = np.concatenate([start.T, end.T], 1)
    coef = np.linspace(1, 0, step).reshape(1, step)
    R = np.concatenate([coef, coef[:, ::-1]], 0)
    return np.dot(L, R).T


def build_step() -> np.ndarray:
    """connect_performing.py 의 addStep(): 한 걸음. (16,6) deg."""
    return np.concatenate([
        linspace(cfg.STANDING_POS, cfg.WALKING_POS1, 2),
        linspace(cfg.WALKING_POS1, cfg.WALKING_POS2, 2),
        linspace(cfg.WALKING_POS2, cfg.WALKING_POS3, 2),
        linspace(cfg.WALKING_POS3, cfg.WALKING_POS4, 2),
        linspace(cfg.WALKING_POS4, cfg.STANDING_POS, 8),
    ], 0)


def build_rise() -> np.ndarray:
    """connect_performing.py 의 addRise(): 굴러서 일어서기. (82,6) deg.

    서 있는 자세에서 다리를 뻗어(EXTENTION) 옆으로 넘어지고(ROLLED),
    누운 자세(SLEEPING)를 거쳐 다시 선다(STANDING).
    """
    return np.concatenate([
        linspace(cfg.STANDING_POS, cfg.EXTENTION_POS, 10),
        linspace(cfg.EXTENTION_POS, cfg.ROLLED_POS, 2),
        linspace(cfg.ROLLED_POS, cfg.SLEEPING_POS, 40),
        linspace(cfg.SLEEPING_POS, cfg.STANDING_POS, 30),
    ], 0)


def build_walk_rise_walk(num_steps: int = 3) -> np.ndarray:
    """걷기 -> 일어서기 -> 걷기. sim 관절 각도(rad) (N,6).

    반환 열 순서는 정책 출력과 같은 인덱스 순서(0..5)라, chair_sim.py 의
    JOINT_ORDERS 매핑을 그대로 태우면 된다.
    """
    deg = np.concatenate(
        [build_step() for _ in range(num_steps)]
        + [build_rise()]
        + [build_step() for _ in range(num_steps)],
        0,
    )
    return real_deg_to_sim_rad(safe_clip(deg))


class KeyframePlayer:
    """키프레임 큐를 한 틱에 한 줄씩 내보내고, 끝나면 마지막 줄을 유지한다.

    실기 src/*.py 의 재생 루프와 같은 동작이다:
        command.data = commands[0, :]
        if (commands.shape[0] - 1): commands = np.delete(commands, 0, 0)
    -- 마지막 한 줄은 지우지 않으므로 큐가 마르면 그 자세로 정지한다.
    """

    def __init__(self, queue: np.ndarray, loop: bool = False):
        assert len(queue) > 0, "빈 큐"
        self.queue = queue
        self.loop = loop
        self.idx = 0
        self.done = False  # loop=True 면 끝나지 않는다
        self.laps = 0
        self._lap_pending = False

    def next(self) -> np.ndarray:
        """이번 틱에 보낼 목표각 한 줄.

        큐를 다 쓰면 loop=True 는 첫 줄로 돌아가고, loop=False 는 마지막 줄을
        계속 돌려준다.
        """
        row = self.queue[self.idx]
        if self.idx + 1 < len(self.queue):
            self.idx += 1
        elif not self.done:  # 마지막 줄을 방금 내보냈다
            self.laps += 1
            self._lap_pending = True
            if self.loop:
                self.idx = 0
            else:
                self.done = True
        return row

    def just_finished(self) -> bool:
        """한 바퀴를 방금 끝냈으면 한 번만 True (완료/반복 로그용)."""
        if self._lap_pending:
            self._lap_pending = False
            return True
        return False


if __name__ == "__main__":
    q = build_walk_rise_walk()
    print(f"{len(q)} rows, {len(q) / 20.0:.1f}s @20Hz, "
          f"|q|max={np.abs(q).max():.4f} rad")
