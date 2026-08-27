"""gym 등록이 Isaac 없이 성립하는지. entry point 가 문자열이므로 import chair_rl 은 Kit 을 띄우지 않는다."""

import gymnasium as gym

import chair_rl  # noqa: F401  — 등록 부수효과


def test_walk_task_registered_lazily():
    spec = gym.spec("Chair-Walk-Direct-v0")
    assert spec.entry_point == "chair_rl.walk_env:WalkEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "chair_rl.walk_env:WalkEnvCfg"
    assert spec.disable_env_checker is True
