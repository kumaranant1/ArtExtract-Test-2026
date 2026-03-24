"""
This is the Main script for Task 1.
This is just modular implementation of the code in task1-CRNN-WikiArt-Classification.ipynb for clarity. 
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from models import CRNN
from utils import WikiArtDataset, Trainer, evaluate
from utils import load_class_names, plot_outliers
from utils.clean_filenames import clean_image_filenames
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None


def main():
    # Configs
    DATA_DIR = "/data/wikiart" # follow the instructions in README to set this up
    TRAIN_CSV = "data/train_metadata.csv"
    VAL_CSV = "data/val_metadata.csv"
    ARTIST_CLASS_PATH = "data/wikiart_csv/artist_class.txt"
    STYLE_CLASS_PATH = "data/wikiart_csv/style_class.txt"
    GENRE_CLASS_PATH = "data/wikiart_csv/genre_class.txt"
    BATCH_SIZE = 128

    # The image files in the WikiArt dataset have some special characters (like apostrophes) that can cause issues during loading, so we will clean the filenames first before proceeding with data loading and training. 
    # This is a one-time cleanup step to ensure all filenames are standardized and won't cause issues later on.
    clean_image_filenames(DATA_DIR)

    # Data loading and Transforms
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224), # since texture, and brush patters are important, we can use random cropping to capture different parts of the image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # for validation center cropping will be adequate
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = WikiArtDataset(csv_file=TRAIN_CSV, root_dir=DATA_DIR, transform=data_transform['train'])
    val_dataset = WikiArtDataset(csv_file=VAL_CSV, root_dir=DATA_DIR, transform=data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # class idx to name mapping
    artist_names = load_class_names(ARTIST_CLASS_PATH)
    style_names = load_class_names(STYLE_CLASS_PATH)
    genre_names = load_class_names(GENRE_CLASS_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CRNN(num_artists=23, num_styles=27, num_genres=10, rnn_hidden_dim=512).to(device)

    # Training Phase 1: Training the RNN & Heads (Freezing CNN)
    print("\n--- Phase 1: Training Starts (ResNet50 Frozen) ---")
    for param in model.cnn.parameters():
        param.requires_grad = False

    optimizer_phase1 = optim.Adam(model.parameters(), lr=1e-4)
    epochs_phase1 = 10

    trainer_phase1 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs_phase1,
        optimizer=optimizer_phase1,
        device=device
    )

    history_phase1 = trainer_phase1.train()

    with open("history_phase1.json", 'w') as f:
        json.dump(history_phase1, f)

    print("\nPhase 1 Training Complete!")
    print(f"Best Baseline Global F1: {trainer_phase1.best_f1:.4f}")

    # Training Phase 2: Fine-Tuning the Whole Network
    print("\n--- Phase 2: Fine-Tuning Starts (ResNet50 Unfrozen) ---")
    for param in model.cnn.parameters():
        param.requires_grad = True

    # use a smaller learning rate for fine-tuning
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5) 
    epochs_phase2 = 10

    trainer_phase2 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs_phase2,
        optimizer=optimizer_phase2,
        device=device,
        previous_best_f1=trainer_phase1.best_f1,
        load=True,
        load_path='MultiHead_CRNN_classifier_BestModel.pth'
    )

    history_phase2 = trainer_phase2.train()

    with open("history_phase2.json", 'w') as f:
        json.dump(history_phase2, f)

    print("\nPhase 2 Training Complete!")
    print(f"Absolute Best Global F1: {trainer_phase2.best_f1:.4f}")

    
    plot_outliers(model, val_loader, val_dataset, device, artist_names, style_names, genre_names)

if __name__ == "__main__":
    main()