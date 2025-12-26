
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Stage:
    name: str # Name of the stage, used for folder/checkpoint naming
    timesteps: int # Total timesteps to train in this stage
    doom_skill: int # Doom difficulty level
    living_reward: float # Reward constant added every step (usually negative to penalize time)
    frame_skip: int = 4
    
    # Reward Shaping Params
    health_penalty: float = 0.0 # Penalty per health point lost
    ammo_penalty: float = 0.0 # Penalty per ammo point used
    
    # Optional override for scenario. If None, uses the curriculum's base scenario
    scenario: Optional[str] = None 

@dataclass
class Curriculum:
    name: str
    scenario: str # Base scenario (e.g. deadly_corridor.cfg)
    stages: List[Stage]

# --- PREDEFINED CURRICULUMS ---

DEADLY_CORRIDOR_CURRICULUM = Curriculum(
    name="deadly_corridor_v5",
    scenario="deadly_corridor.cfg",
    stages=[
        Stage(
            name="skill2_warmup", 
            timesteps=500_000, 
            doom_skill=2, 
            living_reward=-0.01,
            frame_skip=3,
            health_penalty=0.05, # Ease into it
            ammo_penalty=0.0
        ),
        Stage(
            name="skill4_mid", 
            timesteps=1_000_000, 
            doom_skill=4, 
            living_reward=0.0, # Neutral living reward
            frame_skip=3,
            health_penalty=0.1, # Full penalty
            ammo_penalty=0.01 
        ),
        Stage(
            name="skill5_target", 
            timesteps=2_000_000, 
            doom_skill=5, 
            living_reward=0.0,
            frame_skip=2, # More reactive
            health_penalty=0.1,
            ammo_penalty=0.01 
        ),
    ]
)

DEFEND_CENTER_CURRICULUM = Curriculum(
    name="defend_center_v5",
    scenario="defend_the_center.cfg",
    stages=[
        Stage(
            name="skill2_warmup",
            timesteps=400_000,
            doom_skill=2,
            living_reward=-0.005,
            frame_skip=3,
            health_penalty=0.05,
            ammo_penalty=0.01
        ),
        Stage(
            name="skill4_target",
            timesteps=1_500_000,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.05
        )
    ]
)

GRAND_CURRICULUM = Curriculum(
    name="universal_v5",
    scenario="generic.cfg", # Not used directly, stages override
    stages=[
        # Phase 1: Basic Mechanics (Movement + Shooting stationary)
        Stage(
            name="phase1_basic",
            scenario="basic.cfg",
            timesteps=300_000,
            doom_skill=5, # Easy map but fast
            living_reward=-0.01,
            frame_skip=4,
            health_penalty=0.0, 
            ammo_penalty=0.0 # Learn to shoot freely first
        ),
        # Phase 2: Defense (Turning + Ammo Management)
        Stage(
            name="phase2_defend",
            scenario="defend_the_center.cfg",
            timesteps=1_000_000,
            doom_skill=3,
            living_reward=0.0,
            frame_skip=3,
            health_penalty=0.05,
            ammo_penalty=0.05 # Learn control
        ),
        # Phase 3: Navigation & Combat (Everything)
        Stage(
            name="phase3_corridor",
            scenario="deadly_corridor.cfg",
            timesteps=2_000_000,
            doom_skill=4,
            living_reward=0.0,
            frame_skip=2,
            health_penalty=0.1,
            ammo_penalty=0.01
        )
    ]
)
