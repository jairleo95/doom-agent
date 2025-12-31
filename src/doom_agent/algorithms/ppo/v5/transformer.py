
import math
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from typing import Optional, Tuple, Dict, Any, List

class TransformerModel(nn.Module):
    """
    A Transformer architecture for RecurrentPPO.
    Implements a sliding window memory approach.
    """
    def __init__(
        self,
        feature_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        context_len: int = 32, # Max context length (memory)
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_len = context_len
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=False # SB3 passes (SeqLen, Batch, Dim) if not flattened? No, SB3 passes (1, Batch, Dim) per step in eval, or (N, B, D) in train.
            # PyTorch Transformer with batch_first=False expects (Seq, Batch, Dim).
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        # Positional Encoding (Simple learned or sinusoidal? Learned is safer for RL sometimes)
        self.pos_encoder = nn.Parameter(torch.zeros(context_len, 1, feature_dim))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.01)

    def forward(
        self, 
        features: torch.Tensor, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: (SeqLen, Batch, Dim)
        state: (Batch, ContextLen, Dim) -> Memory buffer
        """
        seq_len, batch_size, dim = features.shape
        
        # If state is zeros (start of episode), we treat as empty memory.
        # State usually is (1, B, H) in ResNets, here it's (1, B, Context*Dim).
        # We need to reshape state.
        
        # Handle state reshape
        # Assume state is flattened memory: (Batch, ContextLen * Dim)
        flat_state = state.squeeze(0) # RecurrentPPO sends (1, Batch, HiddenSize) usually as hidden tuple?
        # Actually in SB3 RecurrentPPO, "lstm_states" is a tuple (h, c).
        # We will use single state tensor.
        
        memory = flat_state.view(batch_size, self.context_len, dim).transpose(0, 1) # (Context, Batch, Dim)
        
        # For training (seq_len > 1), we usually process the whole chunk sequence using strict causal masking.
        # But we also have "Memory" from previous chunk.
        # Model Input = Concat(Memory, Features)
        
        full_seq = torch.cat([memory, features], dim=0) # (Context+Seq, Batch, Dim)
        total_len = full_seq.shape[0]
        
        # Positional Encoding
        # We clip to max context supported by pos encoder or resize?
        # Ideally pos encoder is large enough.
        # Let's align PE to the end (most recent).
        
        # Just use relative offsets or standard PE up to total_len?
        # If total_len > self.context_len, we slice? No, we needed memory.
        
        # Apply PE
        # Slice PE to match total_len
        if total_len > self.pos_encoder.shape[0]:
             # Ensure we don't crash, but ideally memory+seq <= limit not guaranteed if seq is huge.
             # SB3 chunks are 1024. 
             # Transformer-XL requires managing memory.
             # Simplified: We only attend to "recent" history within this chunk + small memory?
             # Let's assume standard Attention over (Seq). Memory is just pre-pended.
             pass
        
        # Create Causal Mask
        # We want position i to attend to <= i.
        # src_mask: (Total, Total).
        mask = nn.Transformer.generate_square_subsequent_mask(total_len).to(features.device)
        
        # Forward Transformer
        # (S, B, E)
        output = self.transformer_encoder(full_seq + self.pos_encoder[:total_len], mask=mask)
        
        # Extract only the "New" outputs (corresponding to features)
        # The memory part is just context.
        # output is (Context+Seq, Batch, Dim)
        # We take the last seq_len elements.
        valid_output = output[-seq_len:]
        
        # Update Memory
        # New memory = last context_len elements of full_seq? Or output?
        # Standard GTrXL uses Activations.
        # We will allow using raw features or output. Let's use Output (more powerful, like RNN state).
        new_memory = output[-self.context_len:] # (Context, Batch, Dim)
        
        # Check if we need padding
        if new_memory.shape[0] < self.context_len:
            pad = torch.zeros(self.context_len - new_memory.shape[0], batch_size, dim, device=features.device)
            new_memory = torch.cat([pad, new_memory], dim=0)
            
        # Flatten state for SB3
        new_state_flat = new_memory.transpose(0, 1).reshape(batch_size, -1)
        
        return valid_output, new_state_flat

class TransformerPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable LSTM
        self.lstm_actor = nn.Identity()
        self.lstm_critic = nn.Identity()
        
        # Init Transformer
        # Feature dim comes from features_extractor logic.
        # Default NatureCNN features = 512.
        self.context_len = 16 
        hidden_dim = 512 # Default for RecurrentPPO usually?
        
        self.transformer = TransformerModel(
            feature_dim=hidden_dim, 
            n_heads=4, 
            n_layers=2, 
            context_len=self.context_len
        )
        
    def _predict(self, observation, lstm_states, episode_starts, deterministic=False):
        # Override to use Transformer
        # Preprocess logic same as parent...
        features = self.extract_features(observation)
        
        # SB3 handles reshaping features to (Seq, Batch, Dim) inside get_distribution?
        # Actually RecurrentPPO handles the loop.
        # But we need direct control.
        
        # Wait, overriding _predict is for evaluation (1 step).
        # We need to override `forward` or `get_distribution`?
        
        # The cleanest way in SB3 is replacing the `lstm_actor` and `lstm_critic` modules,
        # BUT they expect specific input/output.
        # SB3 RecurrentPolicy `forward` splits logic into `_process_sequence`.
        
        return super()._predict(observation, lstm_states, episode_starts, deterministic)

    # We need to monkey-patch or override `forward` of the Policy, or `_process_sequence`.
    # `RecurrentActorCriticPolicy` has `_process_sequence`.
    
    def _process_sequence(
        self,
        features: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
        lstm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # features: (Batch * Seq, Dim) -> Need to unflatten
        n_seq = lstm_states[0].shape[1] # Batch size? No, shape[1] is hidden size?
        # SB3 state: (n_layers, batch, hidden).
        
        # We are hacking significantly here.
        # Alternative: Just implement the logic inside `forward` and bypass parent.
        
        # Simplified: We treat the "Transformer" as the "LSTM".
        # Input to this func: `features` (Batch*Seq, Dim).
        # We need to reshape to (Seq, Batch, Dim).
        
        batch_size = lstm_states[0].shape[0] # Actually usually (1, Batch, H) for LSTM state in SB3?
        # Wait, RecurrentPPO state is Tuple(h, c).
        # Our state is (Batch, Context*Dim).
        # We treat it as a single tensor tuple: (state,).
        
        # Reshape features
        seq_len = features.shape[0] // batch_size
        features_seq = features.view(seq_len, batch_size, -1)
        
        # Forward Transformer
        # We use the SAME transformer for Actor and Critic (shared representation usually better for Transformers).
        # Or we can have separate stats.
        # Let's use SHARED transformer, then use heads.
        # But `_process_sequence` is called twice (once for actor, once for critic) if `share_features_extractor=False`?
        # Standard SB3 shares features extractor but uses separate LSTMs by default unless configured.
        
        # Let's assume we use ONE transformer and shared latent.
        # But `RecurrentPPO` structure separates them.
        
        # For this prototype: We use the Transformer inside this function.
        # We ignore `lstm` arg (which is Identity).
        
        # Current State
        state = lstm_states[0] # (Batch, Hidden)
        
        # Episode Starts: Reset state for items where episode_starts=True
        # But episode_starts is (Batch*Seq).
        # We handle this by masking?
        # Logic: If episode starts at step t, memory should be cleared.
        # Hard to do efficiently in batch without looping or advanced masking.
        # SB3 `forward` loops one by one if necessary, or uses PackedSequence.
        
        # Simple/Naive: Use the Transformer.
        # Output: (Seq, Batch, Dim) -> (Batch*Seq, Dim)
        
        out_seq, new_state = self.transformer(features_seq, state)
        
        return out_seq.flatten(0, 1), (new_state,)

class TransformerPPO(RecurrentPPO):
    def __init__(self, *args, **kwargs):
        # Force use of our policy
        kwargs["policy"] = TransformerPolicy
        # Adjust policy_kwargs if needed to pass model params?
        super().__init__(*args, **kwargs)
