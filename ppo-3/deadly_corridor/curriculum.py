import numpy as np


class CurriculumManager:
    def __init__(self, stages):
        self.stages = stages
        self.current_idx = 0
        self.stage_start_episode = 0

    @property
    def current_stage(self):
        return self.stages[self.current_idx]

    def maybe_advance(self, episode_rewards):
        if self.current_idx >= len(self.stages) - 1:
            return False

        stage = self.current_stage
        total_eps = len(episode_rewards)
        stage_eps = total_eps - self.stage_start_episode

        if stage_eps < stage.min_episodes:
            return False

        window = min(stage.window, stage_eps, len(episode_rewards))
        if window <= 0:
            return False

        mean_reward = float(np.mean(episode_rewards[-window:]))
        if mean_reward >= stage.unlock_mean_reward:
            self.current_idx += 1
            self.stage_start_episode = total_eps
            return True
        return False

    def describe_stage(self):
        stage = self.current_stage
        return (
            f"{self.current_idx + 1}/{len(self.stages)} "
            f"{stage.name} (skill={stage.doom_skill}, "
            f"living_reward={stage.living_reward}, "
            f"reward_scale={stage.reward_scale})"
        )
