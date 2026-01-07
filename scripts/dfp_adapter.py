import os
import sys
import numpy as np
import cv2
import itertools as it
from pathlib import Path

# Disable TensorFlow 2 behavior for DFP (2016 champion)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DFPAdapter:
    def __init__(self, model_path=None):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.dfp_dir = self.root_dir / "external" / "dfp"
        
        # Add DFP source to path
        if str(self.dfp_dir) not in sys.path:
            sys.path.append(str(self.dfp_dir))
            
        from DFP.future_predictor_agent_advantage import FuturePredictorAgentAdvantage
        from DFP.util import make_objective_indices_and_coeffs

        # 1. Setup Arguments (matched to D3_battle)
        self.params = {
            'state_imgs_shape': (1, 84, 84), # (channels, height, width)
            'state_meas_shape': (3,),
            'obj_shape': (18,), # 6 temporal * 3 meas
            'num_simulators': 1,
            'meas_for_net': [0, 1, 2],
            'meas_for_manual': [],
            'target_dim': 18,
            'target_names': ['Health', 'Ammo', 'Frags'] * 6,
            'discrete_controls': [0, 1, 2, 3, 4, 5, 6, 7],
            'discrete_controls_manual': [],
            'opposite_button_pairs': [],
            'preprocess_input_images': lambda x: x / 255. - 0.5,
            'preprocess_input_measurements': lambda x: x / 100. - 0.5,
            'preprocess_input_targets': lambda x: x,
            'postprocess_predictions': lambda x: x,
            'objective_coeffs_temporal': [0., 0., 0., 0.5, 0.5, 1.],
            'objective_coeffs_meas': [0.5, 0.5, 1.0],
            'random_exploration_schedule': lambda x: 0,
            'new_memories_per_batch': 0,
            'add_experiences_every': 100000,
            'random_objective_coeffs': False,
            'objective_coeffs_distribution': 'none',
            'conv_params': np.array([(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                    dtype=[('out_channels', int), ('kernel', int), ('stride', int)]),
            'fc_img_params': np.array([(512,)], dtype=[('out_dims', int)]),
            'fc_meas_params': np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)]),
            'fc_joint_params': np.array([(512,), (-1,)], dtype=[('out_dims', int)]),
            'fc_obj_params': None,
            'weight_decay': 0.0,
            'batch_size': 1,
            'init_learning_rate': 0,
            'lr_step_size': 1000,
            'lr_decay_factor': 1.0,
            'adam_beta1': 0.95,
            'adam_epsilon': 1e-4,
            'optimizer': 'Adam',
            'reset_iter_count': False,
            'clip_gradient': 0,
            'checkpoint_dir': str(self.dfp_dir / 'pretrained'),
            'log_dir': '/tmp/dfp_logs',
            'init_model': str(self.dfp_dir / 'pretrained'),
            'model_name': 'predictor.model',
            'model_dir': 'inference',
            'print_err_every': 1000,
            'detailed_summary_every': 1000,
            'checkpoint_every': 1000,
            'test_policy_every': 0,
            'num_steps_per_policy_test': 0,
            'save_param_histograms_every': 0
        }

        # Calculate objective indices and coeffs
        indices, coeffs = make_objective_indices_and_coeffs(
            self.params['objective_coeffs_temporal'], 
            self.params['objective_coeffs_meas']
        )
        self.params['objective_indices'] = indices
        
        # DFP expects the full objective vector in the feed_dict even if not used in some layers
        full_coeffs = np.zeros(self.params['obj_shape'], dtype=np.float32)
        full_coeffs[indices] = coeffs
        self.params['objective_coeffs'] = full_coeffs

        # 2. Build TF Session and Agent
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.agent = FuturePredictorAgentAdvantage(self.sess, self.params)
        
        # 3. Load Model
        if model_path is None:
            model_path = str(self.dfp_dir / 'pretrained')
        
        print(f"Loading DFP model from {model_path}...")
        self.agent.load(model_path)
        
    def preprocess_obs(self, obs_rgb):
        # DFP expects 84x84 Gray
        gray = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        # Add dimension for batch and channel
        return resized[np.newaxis, :, :, np.newaxis] # (1, 84, 84, 1)

    def select_action(self, obs_rgb, health=100, ammo=50, frags=0):
        screen = self.preprocess_obs(obs_rgb)
        measurements = np.array([[health, ammo, frags]], dtype=np.float32)
        
        # Act
        action_id = self.agent.act_net(screen, measurements, self.params['objective_coeffs'])[0]
        
        # Translate to 8-button vector then to 7-button universal
        dfp_vec = self.agent.net_discrete_actions[action_id]
        return self.translate_action(dfp_vec)

    def translate_action(self, dfp_vec):
        # DFP D3 Buttons: [FWD, BWD, R, L, TL, TR, ATK, SPEED]
        # Universal: [FWD, BWD, L, R, TL, TR, ATK]
        universal_vec = [0] * 7
        if dfp_vec[0]: universal_vec[0] = 1 # FWD
        if dfp_vec[1]: universal_vec[1] = 1 # BWD
        if dfp_vec[3]: universal_vec[2] = 1 # L (DFP has R then L)
        if dfp_vec[2]: universal_vec[3] = 1 # R
        if dfp_vec[4]: universal_vec[4] = 1 # TL
        if dfp_vec[5]: universal_vec[5] = 1 # TR
        if dfp_vec[6]: universal_vec[6] = 1 # ATK
        return universal_vec
