"""
Training curriculum definitions for Dreamer V3.

Similar to PPO v5 curriculum.py but adapted for Dreamer V3's episode-based training.
"""

from dataclasses import dataclass
from typing import List, Optional
# from doom_agent.algorithms.dreamer_v3.doom_envs import DoomEnv # Unused import removed


@dataclass
class Stage:
    """Single training stage configuration."""
    name: str  # Name of the stage
    episodes: int  # Number of episodes to train in this stage
    doom_skill: int  # Doom difficulty level (1-5)
    living_reward: float  # Reward per step
    frame_skip: int = 4
    
    # Reward shaping parameters
    health_penalty: float = 0.0
    ammo_penalty: float = 0.0
    frag_bonus: float = 10.0
    
    # Optional scenario override
    scenario: Optional[str] = None


@dataclass
class Curriculum:
    """Multi-stage training curriculum."""
    name: str
    scenario: str  # Base scenario config file
    stages: List[Stage]


# ============================================================================
# Predefined Curricula
# ============================================================================

DEATHMATCH_CURRICULUM = Curriculum(
    name="deathmatch_dreamer_v3",
    scenario="deathmatch.cfg",
    stages=[
        Stage(
            name="skill2_warmup",
            episodes=500,
            doom_skill=2,
            living_reward=0.0,
            frame_skip=4,
            health_penalty=0.05,
            ammo_penalty=0.01,
            frag_bonus=10.0
        ),
        Stage(
            name="skill3_intermediate",
            episodes=1000,
            doom_skill=3,
            living_reward=0.0,
            frame_skip=4,
            health_penalty=0.1,
            ammo_penalty=0.02,
            frag_bonus=15.0
        ),
        Stage(
            name="skill4_advanced",
            episodes=1500,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=3,
            health_penalty=0.1,
            ammo_penalty=0.02,
            frag_bonus=20.0
        ),
        Stage(
            name="skill5_expert",
            episodes=2000,
            doom_skill=5,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.15,
            ammo_penalty=0.03,
            frag_bonus=25.0
        ),
    ]
)


DEADLY_CORRIDOR_CURRICULUM = Curriculum(
    name="deadly_corridor_dreamer_v3",
    scenario="deadly_corridor.cfg",
    stages=[
        Stage(
            name="skill2_warmup",
            episodes=300,
            doom_skill=2,
            living_reward=-0.01,
            frame_skip=3,
            health_penalty=0.05,
            ammo_penalty=0.0
        ),
        Stage(
            name="skill4_mid",
            episodes=700,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=3,
            health_penalty=0.1,
            ammo_penalty=0.01
        ),
        Stage(
            name="skill5_target",
            episodes=1000,
            doom_skill=5,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.01
        ),
    ]
)


DEFEND_CENTER_CURRICULUM = Curriculum(
    name="defend_center_dreamer_v3",
    scenario="defend_the_center.cfg",
    stages=[
        Stage(
            name="skill2_warmup",
            episodes=300,
            doom_skill=2,
            living_reward=-0.005,
            frame_skip=3,
            health_penalty=0.05,
            ammo_penalty=0.01
        ),
        Stage(
            name="skill4_target",
            episodes=700,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.05
        )
    ]
)


GRAND_CURRICULUM = Curriculum(
    name="universal_dreamer_v3",
    scenario="generic.cfg",
    stages=[
        # Phase 1: Basic Mechanics
        Stage(
            name="phase1_basic",
            scenario="basic.cfg",
            episodes=200,
            doom_skill=5,
            living_reward=-0.01,
            frame_skip=4,
            health_penalty=0.0,
            ammo_penalty=0.0
        ),
        # Phase 2: Defense
        Stage(
            name="phase2_defend",
            scenario="defend_the_center.cfg",
            episodes=500,
            doom_skill=3,
            living_reward=0.0,
            frame_skip=3,
            health_penalty=0.05,
            ammo_penalty=0.05
        ),
        # Phase 3: Navigation & Combat
        Stage(
            name="phase3_corridor",
            scenario="deadly_corridor.cfg",
            episodes=1000,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.01
        ),
        # Phase 4: Ultimate Challenge
        Stage(
            name="phase4_deathmatch",
            scenario="deathmatch.cfg",
            episodes=2000,
            doom_skill=5,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.02,
            frag_bonus=20.0
        ),
    ]
)
