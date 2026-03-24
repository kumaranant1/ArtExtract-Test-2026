import torch

def dino_v2(model_name: str = 'dinov2_vits14'):
    """
    Load the DINOv2 vision transformer model.
    """
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    return model