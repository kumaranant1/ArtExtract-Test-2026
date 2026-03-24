"""
This is the Main script for Task 2.
This is just modular implementation of the code in task2-Painting-Similarity.ipynb for clarity. 
"""
import os
import pickle
import random
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataset import NGADataset, SquarePad
from utils.evaluation import evaluate_retrieval
from utils.retrieve_embeddings import extract_features
from models import dino_v2

# Configs
IMAGE_DIR = "data/NGA/images"
OUTPUT_EMBED_DIR = "data/NGA/dinov2_embeddings.pkl"
OBJECTS_CSV = 'data/NGA/objects.csv'
BATCH_SIZE = 128
MODEL_NAME = 'dinov2_vits14'

def search_and_plot(query_idx, index, embeddings, filenames, df_meta, top_k=5):
    """
    Queries the FAISS index and plots the results.
    """
    query_vector = embeddings[query_idx].reshape(1, -1)
    
    # perform the FAISS search
    cos_scores, indices = index.search(query_vector, top_k + 1)
    
    fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
    
    query_path = os.path.join(IMAGE_DIR, filenames[query_idx])
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("QUERY IMAGE", color='blue', fontweight='bold')
    axes[0].axis('off')
    
    for i in range(1, top_k + 1):
        match_idx = indices[0][i]
        match_score = cos_scores[0][i] * 100
        match_filename = filenames[match_idx]
        match_path = os.path.join(IMAGE_DIR, match_filename)
        
        title_text = f"Sim: {match_score:.1f}%"
        
        if df_meta is not None:
            try:
                obj_id = int(match_filename.split('.')[0])
                row = df_meta[df_meta['objectid'] == obj_id]
                if not row.empty:
                    art_title = row['title'].values[0] if 'title' in row.columns else "Unknown Title"
                    title_text += f"\n{art_title}"
            except ValueError:
                pass 
                
        axes[i].imshow(Image.open(match_path))
        axes[i].set_title(title_text, fontsize=9)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")

    data_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = NGADataset(IMAGE_DIR, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # feature extraction
    if not os.path.exists(OUTPUT_EMBED_DIR):
        print(f"Loading Meta DINOv2 ({MODEL_NAME})...")
        model = dino_v2(MODEL_NAME).to(device)
        extract_features(model, dataloader, device, output_dir=OUTPUT_EMBED_DIR)
    else:
        print(f"Found existing embeddings at {OUTPUT_EMBED_DIR}. Skipping extraction.")

    # load embeddings, build index
    print("Loading Embeddings...")
    with open(OUTPUT_EMBED_DIR, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings'].astype('float32')
    filenames = data['filenames']

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"FAISS Index built with {index.ntotal} paintings.")

    try:
        df_meta = pd.read_csv(OBJECTS_CSV, low_memory=False)
        print("Metadata loaded successfully.")
    except Exception as e:
        print(f"Could not load metadata CSV: {e}")
        df_meta = None

    # Visualizations
    print("\nRunning Visual Queries...")
    queries_to_test = [40, 1000, 500, 2201, 3968, 4036]
    for q_idx in queries_to_test:
        if q_idx < len(embeddings):
            search_and_plot(q_idx, index, embeddings, filenames, df_meta)

    print("\nEvaluating Retrieval Confidence...")
    cosines = []
    top_k = 3
    iterations = range(10)
    sample_size = 10
    
    for _ in iterations:
        cosines.append(evaluate_retrieval(index, embeddings, sample_size=sample_size, top_k=top_k))

    plt.figure(figsize=(8, 5))
    plt.title(f"Cosine confidence realization with sample size {sample_size}")
    plt.plot(iterations, cosines, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Top-{top_k} Cosine Confidence (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    print(f"Mean of evaluation curve: {np.mean(cosines):.2f}%")

if __name__ == "__main__":
    main()