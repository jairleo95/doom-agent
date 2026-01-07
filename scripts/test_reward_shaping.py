import cv2
import random
import sys
import time
from pathlib import Path

# Add project root and src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv, deathmatch_actions
from doom_agent.algorithms.dreamer.v3.agent import DreamerV3Agent
try:
    from scripts.arnold_adapter import ArnoldAdapter
    from scripts.dfp_adapter import DFPAdapter
except (ImportError, ModuleNotFoundError):
    from arnold_adapter import ArnoldAdapter
    from dfp_adapter import DFPAdapter
from omegaconf import OmegaConf

def test_shaping(manual=False, agent_path=None, use_arnold=False, use_dfp=False):
    print("--- 🧪 Validador de Reward Shaping (Incentivos) ---")
    
    # 1. Environment Setup
    # We use a higher resolution for the spectator if manual, but agents need 64x64 or 160x120
    # DoomDreamerEnv will handle resizing for us internally if needed.
    env = DoomDreamerEnv(
        scenario='deathmatch.cfg',
        living_reward=-0.01,
        movement_reward=0.02,
        frag_bonus=20.0,
        health_penalty=0.1,
        obs_shape=(64, 64, 3), # Base shape for incentivized training
        window_visible=True,
        frame_skip=1 if manual else 4 
    )
    
    actions = deathmatch_actions()
    obs = env.reset()
    info = {'health': 100, 'ammo': 50, 'frags': 0}
    
    # 2. Agent Initialization
    agent = None
    mode_text = "ALEATORIO"
    osd_color = (150, 150, 150)
    
    if use_arnold:
        mode_text = "ARNOLD (BOSS MODE)"
        osd_color = (0, 0, 255)
        arnold_path = str(root_dir / "external/arnold/pretrained/vizdoom_2017_track2.pth")
        agent = ArnoldAdapter(arnold_path)
        print("MODO: SOTA ARNOLD 2017")
    elif use_dfp:
        mode_text = "INTEL DFP (2016 CHAMP)"
        osd_color = (255, 100, 0)
        agent = DFPAdapter()
        print("MODO: SOTA INTEL DFP 2016")
    elif agent_path:
        mode_text = "DREAMER V3 (IA)"
        osd_color = (0, 255, 255)
        conf_dir = root_dir / "src/doom_agent/algorithms/dreamer/v3/conf"
        base_cfg = OmegaConf.load(conf_dir / "agent/dreamer_v3.yaml")
        hw_cfg = OmegaConf.load(conf_dir / "hardware/rtx3060.yaml")
        cfg = OmegaConf.merge(base_cfg, hw_cfg)
        
        agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_config['num_actions'] = len(actions)
        agent_config['action_dim'] = len(actions)
        
        agent = DreamerV3Agent(**agent_config)
        # Load logic would go here if we had a checkpoint path
        print(f"MODO: INTELIGENCIA ARTIFICIAL ({agent_path})")
    elif manual:
        mode_text = "MANUAL"
        osd_color = (0, 255, 0)
        print("MODO: MANUAL (W/S/A/D/Q/E/Space)")

    # 3. Main Loop
    key_map = {
        'w': 0, 's': 1, 'a': 2, 'd': 3, 'q': 4, 'e': 5, ' ': 6
    }
    
    total_shaped_reward = 0
    is_first = True
    
    print("\nStarting validation loop...")
    print(f"{'Step':<5} | {'Action':<15} | {'Reward':>10} | {'Health':<7} | {'Frags':<5}")
    print("-" * 55)

    try:
        for i in range(1000):
            # Select action
            action_name = "STAY"
            
            if use_arnold or use_dfp:
                # Benchmark agents return universal vectors
                action_vec = agent.select_action(obs, health=info['health'], ammo=info['ammo'], frags=info['frags'])
                obs, reward, done, info = env.step_manual(action_vec)
                
                # Active buttons names [FWD, BWD, L, R, TL, TR, ATK]
                btn_names = ["FWD", "BWD", "L", "R", "TL", "TR", "ATK"]
                active_btns = [btn_names[idx] for idx, val in enumerate(action_vec) if val == 1]
                action_name = "+".join(active_btns) if active_btns else "STAY"
            elif agent:
                # Dreamer V3 returns action index
                action_idx = agent.select_action(obs, eval_mode=True, is_first=is_first)
                is_first = False
                obs, reward, done, info = env.step(action_idx)
                action_name = actions[action_idx]
            elif manual:
                key = cv2.waitKey(20) & 0xFF
                char = chr(key).lower() if key < 256 else ""
                if key == 27: break # ESC
                
                if char in key_map:
                    action_idx = key_map[char]
                    obs, reward, done, info = env.step(action_idx)
                    action_name = actions[action_idx]
                else:
                    obs, reward, done, info = env.step_manual([0]*7)
                    action_name = "STAY"
            else:
                action_idx = random.randint(0, len(actions)-1)
                obs, reward, done, info = env.step(action_idx)
                action_name = actions[action_idx]

            total_shaped_reward += reward
            
            # Console Log
            if i % 10 == 0 or abs(reward) > 0.05:
                print(f"{i:<5} | {action_name:<15} | {reward:>10.4f} | {info['health']:<7.1f} | {info['frags']:<5.0f}")
            
            # OSD Visualization
            img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            h, w, _ = img.shape
            cv2.rectangle(img, (0, h-65), (w, h), (0,0,0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(img, f"MODO: {mode_text}", (10, h-40), font, 0.6, osd_color, 2)
            cv2.putText(img, f"RECOMPENSA SHAPED: {total_shaped_reward:.2f}", (10, h-15), font, 0.5, (255, 255, 255), 1)
            
            if manual:
                cv2.putText(img, "W/S/A/D: Move  Q/E: Turn  SPACE: Shoot", (w-320, h-15), font, 0.4, (200, 200, 200), 1)

            cv2.imshow("Reward Shaping Validator", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if done:
                print(f"--- Episodio Terminado. Recompensa Total: {total_shaped_reward:.2f} ---")
                obs = env.reset()
                total_shaped_reward = 0
                is_first = True
                if use_arnold: agent.reset()

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--agent_run", type=str)
    parser.add_argument("--arnold", action="store_true")
    parser.add_argument("--dfp", action="store_true")
    args = parser.parse_args()
    
    test_shaping(
        manual=args.manual, 
        agent_path=args.agent_run, 
        use_arnold=args.arnold, 
        use_dfp=args.dfp
    )
