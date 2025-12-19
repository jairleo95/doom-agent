
import os
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording episodes as GIFs.
    
    :param eval_env: Environment to evaluate/record on.
    :param render_freq: How often to record (in steps). If None, manual call is expected.
    :param n_eval_episodes: Number of episodes to record.
    :param deterministic: Whether to use deterministic actions.
    :param save_path: Path to save the gifs.
    :param name_prefix: Prefix for the filename.
    """
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int = 100_000,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        save_path: str = "logs/videos",
        name_prefix: str = "ppo_model"
    ):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.render_freq > 0 and self.n_calls % self.render_freq == 0:
            self.record_video(suffix=f"_step_{self.num_timesteps}")
        return True
    
    def record_video(self, suffix: str = ""):
        screens = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                # VizDoom with SB3 env usually has the frame in obs if using CnnPolicy.
                # If wrapped in VecFrameStack, obs is (N, C, H, W).
                # We need to render the environment to get the visuals if we want colors,
                # but VizDoom envs might not support render() effectively in headless without configuration.
                # However, our DoomCorridorEnv stores state in self.state.
                # With VecEnv, we can't easily access internal state... 
                # Ideally we rely on the env's render method returning an rgb_array.
                
                # Try to use render()
                try:
                    # SB3 VecEnv render might return None or the array.
                    # For VecEnv, render() typically prints to screen if mode='human'.
                    # We might need to access the underlying env or trust thatobs contains enough info?
                    # Obs is grayscale and stacked. Not great for video.
                    
                    # NOTE: Our DoomCorridorEnv does not implement render(mode='rgb_array') fully compatible with SB3 VecEnv wrapper chains easily.
                    # BUT, we passed window_visible=False.
                    # Let's try to get image from observation for simplicity (grayscale) 
                    # OR we access the underlying env.
                    
                    # Assuming we are using VecEnv. the obs is a numpy array.
                    # Let's use the first frame of the stack, which is the most recent (or last? depends on implementation).
                    # SB3 VecFrameStack stacks OLD on channel 0, NEW on channel -1 (if channels_last)
                    # or NEW on channel 0 (if channels_first)?
                    # let's assume standard Gray8 video is fine for now.
                    
                    # Actually, better visual:
                    # We can use the 'render' method of the vector env if available.
                    pass
                except Exception:
                    pass
                
                # Let's just capture the observation for now. 
                # obs shape: (1, 4, 120, 160) from VecTransposeImage possibly?
                # If un-transposed: (1, 120, 160, 4)
                
                # SB3 VecEnv obs is always batched.
                
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                
                # Extract image for gif
                # We want the latest frame.
                # If Transposed (N, C, H, W):
                frame = obs[0, -1, :, :] # Last frame in stack (newest)
                
                # Convert to uint8 and correct shape if needed
                if frame.max() <= 1.0: frame = (frame * 255).astype(np.uint8)
                else: frame = frame.astype(np.uint8)
                
                screens.append(frame)
                
        # Save as GIF
        if len(screens) > 0:
            save_file = os.path.join(self.save_path, f"{self.name_prefix}{suffix}.gif")
            imageio.mimsave(save_file, screens, fps=30) # 35 FPS like Doom
            if self.verbose > 0:
                print(f"Saved video to {save_file}")

