import os
import imageio
import numpy as np
import cv2
import pytorch_lightning as pl

class VideoRecorderCallback(pl.Callback):
    """
    Registra episodios en GIF, inspirado en la implementación de PPO v5.
    Se dispara por frecuencia de pasos (global_step) y no por episodio.
    """

    def __init__(
        self,
        eval_env,
        save_path: str = "videos",
        name_prefix: str = "dreamer_v3",
        render_freq: int = 100_000,  # pasos globales entre grabaciones
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        fps: int = 35,
    ):
        """
        Args:
            eval_env: Environment to grab frames from.
            save_path: Directory to save videos.
            name_prefix: Prefix for video filenames.
            render_freq: Steps between recordings (use 0/None to disable).
            n_eval_episodes: Episodes to record per trigger.
            deterministic: Whether to use deterministic actions.
            fps: Frames per second in the saved GIF.
        """
        super().__init__()
        self.eval_env = eval_env
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.fps = fps

        os.makedirs(save_path, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = getattr(pl_module, 'global_step_custom', trainer.global_step)
        if self.render_freq > 0 and global_step % self.render_freq == 0:
            self.record_video(pl_module, suffix=f"_step_{global_step}")

    def _obs_to_frame(self, obs: np.ndarray) -> np.ndarray:
        # Use cached high-res render if available, otherwise fallback to observation
        if hasattr(self.eval_env, 'last_high_res_render') and self.eval_env.last_high_res_render is not None:
            return self.eval_env.last_high_res_render.copy()
            
        if obs is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
            
        frame = obs
        if frame.ndim == 4:  # batch dim
            frame = frame[0]
            
        # Ensure RGB format for imageio
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        return frame

    def record_video(self, pl_module, suffix: str):
        frames = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            pl_module.reset_state()
            done = False

            while not done:
                frames.append(self._obs_to_frame(obs))
                action = pl_module.select_action(obs, eval_mode=True)
                obs, reward, done = self.eval_env.step(action)

        if frames:
            save_file = os.path.join(self.save_path, f"{self.name_prefix}{suffix}.gif")
            # use duration instead of fps to avoid DeprecationWarning (duration is in ms)
            imageio.mimsave(save_file, frames, duration=1000/self.fps)
            print(f"  Saved video: {save_file}")
