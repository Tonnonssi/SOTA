"""chair_rl — Chair-type tripedal robot, Isaac Lab RL env (설계문서 §1).

import 만으로 gym 태스크가 등록된다. entry point 는 문자열이라 여기서는 Isaac 을 import 하지
않는다 — obs_layout/mdp/mass_spec 의 CPU 테스트가 Kit 없이 계속 돈다 (§9.1).
agent cfg(rsl_rl/rl_games) entry point 는 학습 이슈에서 건다 (§9.5 축소, 이슈 #12).
"""

import gymnasium as gym

gym.register(
    id="Chair-Walk-Direct-v0",
    entry_point="chair_rl.walk_env:WalkEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "chair_rl.walk_env:WalkEnvCfg"},
)
