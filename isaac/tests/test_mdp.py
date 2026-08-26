"""mdp: 논문 Table III/IV 를 옮긴 순수 함수의 해석적 검증.

회전 헬퍼는 실기 src/utils.py 의 quat_rotate/get_basis_vector 와 같은 결과를 내야
한다(같은 (x,y,z,w) 규약). src/utils.py 는 torch·numpy 만 의존해 그대로 import 한다.
"""

import importlib.util
import math
import os

import numpy as np
import pytest
import torch

from chair_rl import mdp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("chair_real_utils", os.path.join(REPO, "src", "utils.py"))
real_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(real_utils)


def quat_xyzw(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return torch.tensor([*(math.sin(h) * axis), math.cos(h)], dtype=torch.float64)


IDENT = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)


# ---------- 회전 헬퍼: 실기 utils 와 동일 ----------

def test_quat_rotate_matches_real_utils():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(64, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    v = torch.randn(64, 3, generator=g, dtype=torch.float64)
    torch.testing.assert_close(mdp.quat_rotate(q, v), real_utils.quat_rotate(q, v), atol=1e-12, rtol=0)


def test_up_proj_matches_real_utils_and_analytic():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(64, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    e_z = torch.tensor([[0., 0., 1.]], dtype=torch.float64).expand(64, 3)
    torch.testing.assert_close(mdp.up_proj(q), real_utils.get_basis_vector(q, e_z)[:, 2], atol=1e-12, rtol=0)
    assert mdp.up_proj(IDENT[None]).item() == pytest.approx(1.0)
    assert mdp.up_proj(quat_xyzw([1, 0, 0], 90)[None]).item() == pytest.approx(0.0, abs=1e-12)
    assert mdp.up_proj(quat_xyzw([1, 0, 0], 180)[None]).item() == pytest.approx(-1.0)


def test_heading_proj_and_reward():
    pos = torch.zeros(1, 3, dtype=torch.float64)
    tgt = torch.tensor(mdp.P_TARGET, dtype=torch.float64)
    assert mdp.heading_proj(IDENT[None], pos, tgt).item() == pytest.approx(1.0)
    assert mdp.heading_proj(quat_xyzw([0, 0, 1], 90)[None], pos, tgt).item() == pytest.approx(0.0, abs=1e-12)
    # 논문: min{1, (1/0.8)·hp}. hp=1 -> 1 (클립), hp=0 -> 0, hp=-1 -> -1.25 (아래로는 클립 없음)
    hp = torch.tensor([1.0, 0.0, -1.0, 0.4], dtype=torch.float64)
    torch.testing.assert_close(mdp.heading_reward(hp), torch.tensor([1.0, 0.0, -1.25, 0.5], dtype=torch.float64))


# ---------- 보상 항 ----------

def test_progress_is_positive_when_moving_toward_target():
    dt = 0.1
    tgt = torch.tensor(mdp.P_TARGET, dtype=torch.float64)
    p0 = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)
    pot = mdp.potential(p0, tgt, dt)
    assert pot.item() == pytest.approx(-math.sqrt(10.0**2 + 0.1**2) / dt)
    r, pot2 = mdp.progress(pot, p0 + torch.tensor([[0.01, 0., 0.]], dtype=torch.float64), tgt, dt)
    assert r.item() == pytest.approx(0.01 / dt, rel=1e-3)   # 1 cm 전진 / 0.1 s = +0.1
    r_back, _ = mdp.progress(pot, p0 - torch.tensor([[0.01, 0., 0.]], dtype=torch.float64), tgt, dt)
    assert r_back.item() < 0
    assert pot2.item() > pot.item()


def test_height_and_up_rewards_clip_at_one():
    z = torch.tensor([[0., 0., 0.101], [0., 0., 0.04], [0., 0., 0.2]], dtype=torch.float64)
    torch.testing.assert_close(mdp.height_reward(z), torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64))
    up = torch.tensor([1.0, 0.93, 0.465, -0.5], dtype=torch.float64)
    torch.testing.assert_close(mdp.up_reward(up), torch.tensor([1.0, 1.0, 0.5, -0.5 / 0.93], dtype=torch.float64))


def test_action_and_vel_costs():
    a = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float64)
    a_prev = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5, 0.9]], dtype=torch.float64)
    assert mdp.action_cost(a, a_prev).item() == pytest.approx(0.01 + 0.09)
    w = torch.full((1, 6), mdp.OMEGA_MAX - mdp.OMEGA_TOL, dtype=torch.float64)   # 각 관절이 정규화 한계 속도
    assert mdp.vel_cost(w).item() == pytest.approx(6.0)
    assert mdp.vel_cost(torch.zeros(1, 6, dtype=torch.float64)).item() == 0.0


def test_walk_total_weights_and_death_overwrite():
    W = mdp.WalkRewardWeights()
    assert (W.progress, W.height, W.up, W.heading, W.alive, W.death, W.action, W.vel) == (30, 20, 5, 2, 1, -1, -2, -2)
    terms = {"progress": torch.tensor([0.1, 0.1]), "height": torch.tensor([1.0, 1.0]),
             "up": torch.tensor([1.0, 1.0]), "heading": torch.tensor([1.0, 1.0]),
             "action": torch.tensor([0.0, 0.0]), "vel": torch.tensor([0.0, 0.0])}
    terminated = torch.tensor([False, True])
    total = mdp.walk_total(terms, W, terminated)
    assert total[0].item() == pytest.approx(30 * 0.1 + 20 + 5 + 2 + 1)
    assert total[1].item() == pytest.approx(-1.0)   # 논문: death 는 이전 보상을 "덮어쓴다"


def test_walk_reward_terms_shapes_and_keys():
    N = 5
    pos = torch.tensor([[0., 0., 0.1]] * N, dtype=torch.float64)
    q = IDENT.expand(N, 4).clone()
    tgt = torch.tensor(mdp.P_TARGET, dtype=torch.float64)
    pot = mdp.potential(pos, tgt, 0.1)
    a = torch.zeros(N, 6, dtype=torch.float64); jv = torch.zeros(N, 6, dtype=torch.float64)
    terms, pot2 = mdp.walk_reward_terms(pos, q, pot, a, a, jv, 0.1, tgt)
    assert set(terms) == {"progress", "height", "up", "heading", "action", "vel"}
    assert all(t.shape == (N,) for t in terms.values()) and pot2.shape == (N,)
    assert terms["up"][0].item() == pytest.approx(1.0) and terms["height"][0].item() == pytest.approx(1.0)
