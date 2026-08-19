import stable_retro
import gymnasium as gym

class SonicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_info = {}
        self.prev_score = 0
        self.in_special_stage = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_info = info
        self.prev_score = 0
        self.in_special_stage = False
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(info)
        self.prev_info = info
        self.prev_score = info['score']
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info):
        reward = 0

        # Score delta — covers rings, enemies, general progress
        score_delta = info['score'] - self.prev_score
        reward += score_delta * 0.01

        # Special stage transitions
        entered_special = (info['special_stage'] == 1 and not self.in_special_stage)
        exited_special = (info['special_stage'] == 0 and self.in_special_stage)

        if entered_special:
            reward += 100
            self.in_special_stage = True

        if exited_special:
            score_increased = info['score'] > self.prev_score
            if score_increased:
                reward += 500   # Emerald collected
            else:
                reward -= 200   # Failed special stage
            self.in_special_stage = False

        # Act completion
        if info['level_end_bonus'] > 0:
            reward += 300

        # Death penalty
        if info['lives'] < self.prev_info.get('lives', 3):
            reward -= 300

        # Small time penalty to discourage standing still
        reward -= 0.1

        return reward


def make_env():
    env = stable_retro.make("SonicAndKnuckles3-Genesis-v0")
    env = SonicRewardWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("Reward wrapper working!")
    print(f"Observation shape: {obs.shape}")

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if reward != -0.1:
            print(f"Step {i+1} | Reward: {reward:.2f} | Info: {info}")

    env.close()
