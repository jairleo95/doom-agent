import numpy as np
import random

class RandomAdapter:
    """
    Control baseline agent that takes uniform random actions.
    Useful for establishing the 'floor' performance.
    """
    def __init__(self):
        pass # No setup needed

    def select_action(self, obs, health=100, ammo=50, frags=0):
        # Universal Action Vector: [FWD, BWD, L, R, TL, TR, ATK]
        # Randomly choose one button to press, or a combination
        # For a stronger baseline, we bias towards FWD+ATK/TR
        
        action = [0] * 7
        
        # 20% chance to attack
        if random.random() < 0.2:
            action[6] = 1
            
        # 50% chance to move/turn
        if random.random() < 0.5:
            move_idx = random.randint(0, 5)
            action[move_idx] = 1
            
        return action

    def reset(self):
        pass
