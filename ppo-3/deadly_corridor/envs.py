import torch.multiprocessing as mp
from collections import deque

import cv2
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable

from .config import FRAME_STACK, IMG_SIZE, StageConfig, ShapingConfig, SHAPING_CFG


def create_doom_game(
    config_path="deadly_corridor.cfg",
    visible=False,
    doom_skill=5,
    living_reward=0.0,
):
    game = DoomGame()
    game.load_config(config_path)
    if doom_skill is not None:
        game.set_doom_skill(int(doom_skill))
    if living_reward is not None:
        game.set_living_reward(float(living_reward))
    game.set_window_visible(visible)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.init()
    return game


def get_deadly_corridor_actions():
    # [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
    return [
        [0, 0, 0, 1, 0, 0, 0],  # forward
        [0, 0, 1, 0, 0, 0, 0],  # attack
        [0, 0, 1, 1, 0, 0, 0],  # forward + attack
        [0, 0, 0, 0, 0, 1, 0],  # turn left
        [0, 0, 0, 0, 0, 0, 1],  # turn right
        [0, 0, 1, 0, 0, 1, 0],  # turn left + attack
        [0, 0, 1, 0, 0, 0, 1],  # turn right + attack
        [1, 0, 0, 0, 0, 0, 0],  # strafe left
        [0, 1, 0, 0, 0, 0, 0],  # strafe right
        [0, 0, 0, 0, 1, 0, 0],  # backward
        [0, 0, 0, 1, 0, 1, 0],  # forward + turn left
        [0, 0, 0, 1, 0, 0, 1],  # forward + turn right
    ]


def preprocess_frame(frame, new_size=IMG_SIZE):
    if frame.ndim == 3 and frame.shape[0] <= 4:
        frame = np.transpose(frame, (1, 2, 0))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, :, :]


def init_frame_stack(game):
    state = game.get_state()
    frame = preprocess_frame(state.screen_buffer)
    stack = deque([frame for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    return stack_to_state(stack), stack


def stack_to_state(stack):
    return np.concatenate(list(stack), axis=0)


def update_frame_stack(stack, game):
    frame = preprocess_frame(game.get_state().screen_buffer)
    stack.append(frame)
    return stack_to_state(stack), stack


def _init_env_with_stage(stage: StageConfig):
    g = create_doom_game(
        "deadly_corridor.cfg",
        visible=False,
        doom_skill=stage.doom_skill,
        living_reward=stage.living_reward,
    )
    g.new_episode()
    s, stack = init_frame_stack(g)
    return g, s, stack


def _worker(remote, parent_remote, stage: StageConfig, shaping: ShapingConfig):
    parent_remote.close()
    actions = get_deadly_corridor_actions()
    game, state, frame_stack = _init_env_with_stage(stage)

    prev_kills = 0.0
    prev_ammo = 0.0
    prev_health = 0.0
    attack_cd = 0
    episode_raw_reward = 0.0

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            action_idx = int(data)
            action = actions[action_idx]

            reward_env = game.make_action(action)
            done = game.is_episode_finished()
            episode_raw_reward += reward_env

            kills = game.get_game_variable(GameVariable.KILLCOUNT) if not done else prev_kills
            ammo = game.get_game_variable(GameVariable.AMMO2) if not done else prev_ammo
            health = game.get_game_variable(GameVariable.HEALTH) if not done else prev_health

            delta_kill = max(0.0, kills - prev_kills)
            delta_ammo = ammo - prev_ammo
            delta_health = health - prev_health

            shaping_reward = (
                shaping.reward_kill * delta_kill
                + shaping.reward_health_scale * delta_health
                + shaping.reward_ammo_scale * delta_ammo
            )

            if action[2] == 1 and attack_cd > 0:
                shaping_reward -= shaping.attack_spam_penalty
            attack_cd = shaping.attack_cooldown if action[2] == 1 else max(attack_cd - 1, 0)

            prev_kills, prev_ammo, prev_health = kills, ammo, health

            reward = (reward_env + shaping_reward) * stage.reward_scale

            info = {}
            if done:
                info["episode_reward"] = episode_raw_reward
                episode_raw_reward = 0.0
                game.new_episode()
                state, frame_stack = init_frame_stack(game)
                prev_kills = prev_ammo = prev_health = 0.0
                attack_cd = 0
            else:
                state, frame_stack = update_frame_stack(frame_stack, game)

            remote.send((state, reward, done, info))

        elif cmd == "reset":
            game.close()
            game, state, frame_stack = _init_env_with_stage(stage)
            prev_kills = prev_ammo = prev_health = 0.0
            attack_cd = 0
            episode_raw_reward = 0.0
            remote.send(state)

        elif cmd == "set_stage":
            stage = data
            game.close()
            game, state, frame_stack = _init_env_with_stage(stage)
            prev_kills = prev_ammo = prev_health = 0.0
            attack_cd = 0
            episode_raw_reward = 0.0
            remote.send(state)

        elif cmd == "close":
            game.close()
            remote.close()
            break
        else:
            raise NotImplementedError(cmd)


class ParallelEnv:
    def __init__(self, n_envs: int, stage: StageConfig, shaping: ShapingConfig = SHAPING_CFG):
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.ps = [
            ctx.Process(target=_worker, args=(work_remote, remote, stage, shaping))
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for work_remote in self.work_remotes:
            work_remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, infos = zip(*results)
        return (
            np.stack(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        states = [remote.recv() for remote in self.remotes]
        return np.stack(states)

    def set_stage(self, stage: StageConfig):
        for remote in self.remotes:
            remote.send(("set_stage", stage))
        states = [remote.recv() for remote in self.remotes]
        return np.stack(states)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
