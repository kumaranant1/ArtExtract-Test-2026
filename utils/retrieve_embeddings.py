import torch
from tqdm import tqdm
import pickle
import numpy as np

@torch.no_grad()
def extract_features(model, dataloader, device, output_dir="embeddings.pkl"):
    """
    Extracts features from the given model and dataloader, and saves them to a file.
    """
    all_embeddings = []
    all_filenames = []
    
    model.eval()
    for images, filenames in tqdm(dataloader, desc="Extracting Features"):
        images = images.to(device)
        features = model(images)
        all_embeddings.append(features.cpu().numpy())
        all_filenames.extend(filenames)
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"Extraction complete! embedded matrix shape: {all_embeddings.shape}")
    
    with open(output_dir, 'wb') as f:
        pickle.dump({'embeddings': all_embeddings, 'filenames': all_filenames}, f)
    
    print(f"Extracted features saved to {output_dir}")
