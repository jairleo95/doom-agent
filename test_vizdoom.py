from vizdoom import DoomGame
import random
import time

game = DoomGame()
game.load_config("basic.cfg")  # Viene incluido en ViZDoom
game.init()

actions = [
    [1, 0, 0],  # Moverse a la izquierda
    [0, 1, 0],  # Moverse a la derecha
    [0, 0, 1],  # Disparar
]

episodes = 10

for i in range(episodes):
    print("Episode:", i + 1)
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        r = game.make_action(random.choice(actions))
        time.sleep(0.02)

    print("Reward:", game.get_total_reward())

game.close()
