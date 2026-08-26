#!/usr/bin/env python3
"""isaac/keyframes.py 단위 테스트. Isaac Sim 없이 numpy만으로 돈다.

    ~/miniconda3/envs/RL/bin/python isaac/test_keyframes.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import keyframes as kf  # noqa: E402

# chair_sim.py 가 손으로 적어둔 STANDING 자세(sim rad). 변환 함수가 이 값을
# 재현해야 한다 -- 정책 경로와 스크립트 경로가 같은 좌표계를 쓴다는 뜻.
STANDING_SIM = [-0.17453, 0.0, -0.17453, 0.0, 0.17453, 0.0]

JOINT_LIMIT = 0.872665  # MJCF joint range, rad


def test_standing_roundtrip():
    """STANDING_POS(실기 deg) -> sim rad 가 chair_sim.STANDING_SIM 과 일치한다."""
    got = kf.real_deg_to_sim_rad(kf.cfg.STANDING_POS)[0]
    assert np.allclose(got, STANDING_SIM, atol=1e-4), f"{got} != {STANDING_SIM}"


def test_conversion_is_inverse_of_sim_rad_to_real_deg():
    """src/rl_walk.py 의 simRad2realDeg() 와 정확히 역함수 관계다."""
    rng = np.random.default_rng(0)
    sim_rad = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(1, 6))

    # src/rl_walk.py:57 simRad2realDeg() 를 그대로 옮긴 것
    deg = -np.rad2deg(sim_rad) + 90
    real_deg = np.zeros([1, 6])
    for i in range(6):
        real_deg[0, i] = deg[0, 5 - i]

    back = kf.real_deg_to_sim_rad(real_deg)
    assert np.allclose(back, sim_rad, atol=1e-9), f"{back} != {sim_rad}"


def test_safe_clip_matches_real_robot():
    """safe_clip 이 src 의 safeClip() 과 같다: [40,140] 클립 + 정수 변환.

    ROLLED_POS 에는 150 이 들어 있어 실기에서도 140 으로 잘린다. 자르지 않으면
    관절 한계(±50도)를 넘는 목표각이 나온다.
    """
    assert kf.cfg.ROLLED_POS.max() == 150, "전제가 깨졌다: ROLLED_POS 에 150 이 없다"
    clipped = kf.safe_clip(kf.cfg.ROLLED_POS)
    assert clipped.max() == 140
    assert clipped.min() >= 40
    assert clipped.dtype.kind in "iu", f"정수여야 한다: {clipped.dtype}"

    frac = kf.safe_clip(np.array([[90.7, 80.2, 90.9, 100.5, 90.1, 100.8]]))
    assert (frac[0] == [90, 80, 90, 100, 90, 100]).all(), frac


def test_all_keyframes_within_joint_limit():
    """재생 큐 전체가 MJCF 관절 한계 안에 있다."""
    q = kf.build_walk_rise_walk()
    assert np.abs(q).max() <= JOINT_LIMIT + 1e-9, f"max={np.abs(q).max()}"


def test_queue_length_matches_real_sequence():
    """큐 길이가 실기 시퀀스의 스텝 수 합과 같다.

    addStep = 2+2+2+2+8 = 16, addRise = 10+2+40+30 = 82 (connect_performing.py 기준)
    """
    assert len(kf.build_step()) == 16
    assert len(kf.build_rise()) == 82
    assert len(kf.build_walk_rise_walk()) == 3 * 16 + 82 + 3 * 16


def test_queue_starts_and_ends_at_standing():
    """큐는 STANDING 에서 출발해 STANDING 으로 끝난다 (warmup 자세와 연속)."""
    q = kf.build_walk_rise_walk()
    assert np.allclose(q[0], STANDING_SIM, atol=1e-2), q[0]
    assert np.allclose(q[-1], STANDING_SIM, atol=1e-2), q[-1]


def test_player_advances_one_row_per_tick():
    q = np.arange(12, dtype=float).reshape(4, 3)
    player = kf.KeyframePlayer(q)
    assert (player.next() == q[0]).all()
    assert (player.next() == q[1]).all()
    assert (player.next() == q[2]).all()
    assert not player.done


def test_player_holds_last_row_forever():
    """큐가 끝나면 마지막 자세를 계속 유지한다(실기의 hold 동작). 범위를 넘지 않는다."""
    q = np.arange(6, dtype=float).reshape(2, 3)
    player = kf.KeyframePlayer(q)
    player.next()
    assert not player.done
    assert (player.next() == q[-1]).all()
    assert player.done, "마지막 줄을 낸 시점에 done 이어야 한다"
    for _ in range(100):  # 넘겨도 IndexError 가 나면 안 된다
        assert (player.next() == q[-1]).all()


def test_player_done_latches_once():
    """done 은 한 번만 새로 True 가 된다(완료 로그를 한 번만 찍기 위해)."""
    q = np.zeros((2, 3))
    player = kf.KeyframePlayer(q)
    player.next()
    assert player.just_finished() is False
    player.next()
    assert player.just_finished() is True
    assert player.just_finished() is False


def test_player_loops_back_to_start():
    """loop=True 면 마지막 줄 다음에 첫 줄로 돌아간다."""
    q = np.arange(9, dtype=float).reshape(3, 3)
    player = kf.KeyframePlayer(q, loop=True)
    assert (player.next() == q[0]).all()
    assert (player.next() == q[1]).all()
    assert (player.next() == q[2]).all()
    assert (player.next() == q[0]).all(), "첫 줄로 돌아와야 한다"
    assert (player.next() == q[1]).all()


def test_player_loop_never_done():
    q = np.zeros((3, 3))
    player = kf.KeyframePlayer(q, loop=True)
    for _ in range(50):
        player.next()
    assert not player.done
    assert player.laps == 16, f"3줄짜리 큐를 50틱 = 16바퀴, got {player.laps}"


def test_player_loop_reports_each_lap_once():
    """just_finished() 는 한 바퀴에 정확히 한 번만 True (반복 로그용)."""
    q = np.zeros((2, 3))
    player = kf.KeyframePlayer(q, loop=True)
    trues = 0
    for _ in range(10):
        player.next()
        if player.just_finished():
            trues += 1
    assert trues == 5, f"2줄짜리 큐를 10틱 = 5바퀴, got {trues}"


def test_player_no_loop_reports_once_only():
    """loop=False 는 끝난 뒤 계속 next() 해도 완료 보고가 한 번뿐이다."""
    q = np.zeros((2, 3))
    player = kf.KeyframePlayer(q)
    trues = 0
    for _ in range(20):
        player.next()
        if player.just_finished():
            trues += 1
    assert trues == 1, f"got {trues}"
    assert player.laps == 1, f"got {player.laps}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failed)")
    sys.exit(1 if fails else 0)
