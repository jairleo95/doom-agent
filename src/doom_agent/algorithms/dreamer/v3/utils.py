import torch

def flip_actions(actions_tensor):
    """
    Flip horizontal actions for data augmentation.
    Universal set indices: 3:TL, 4:TR, 5:TL+ATK, 6:TR+ATK, 7:SL, 8:SR, 10:FWD+TL, 11:FWD+TR
    """
    flip_map = {3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 10: 11, 11: 10}
    
    # Ensure we don't index out of bounds
    max_act = int(actions_tensor.max().item()) if actions_tensor.numel() > 0 else 12
    lookup = torch.arange(max(max_act + 1, 12), device=actions_tensor.device)
    
    for k, v in flip_map.items(): 
        if k < len(lookup):
            lookup[k] = v
            
    return lookup[actions_tensor]

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"
