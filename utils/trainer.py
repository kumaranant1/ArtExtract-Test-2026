import torch
import torch.nn as nn
from tqdm import tqdm
from .evaluation import evaluate

class Trainer:
    """
    Trainer class to handle the training loop, validation, and model checkpointing based on the global F1 score. 
    """
    def __init__(self, model, train_loader, val_loader, num_epochs, optimizer, device, previous_best_f1=0.0, load=False, load_path=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        
        if load:
            if load_path is None:
                raise ValueError("[ERROR] please provide a load_path!")
            print(f"[INFO] Loading pre-trained weights from '{load_path}'...")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print("[INFO] Weights loaded successfully!")
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_f1': [],
            'style_top1': [], 'artist_top1': [], 'genre_top1': [],
            'style_top5': [], 'artist_top5': [], 'genre_top5': [],
            'style_f1': [], 'artist_f1': [], 'genre_f1': []
        }

        self.best_f1 = previous_best_f1

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for images, artists, styles, genres in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} : ", leave=True):
            images, artists, styles, genres = images.to(self.device), artists.to(self.device), styles.to(self.device), genres.to(self.device)

            self.optimizer.zero_grad()
            out_artist, out_style, out_genre = self.model(images)

            loss_a = self.criterion(out_artist, artists)
            loss_s = self.criterion(out_style, styles)
            loss_g = self.criterion(out_genre, genres)

            # combining the loss
            total_loss = loss_a + loss_s + loss_g
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

        return running_loss / len(self.train_loader)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):

            # do one epoch
            train_loss = self.train_one_epoch(epoch)

            # evaluate the model
            metrics = evaluate(self.model, self.val_loader, self.criterion, self.device)

            # log the metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(metrics['val_loss'])
            self.history['val_f1'].append(metrics['global_f1'])
            
            self.history['style_top1'].append(metrics['style']['top1'])
            self.history['style_top5'].append(metrics['style']['top5'])
            self.history['style_f1'].append(metrics['style']['f1'])

            self.history['artist_top1'].append(metrics['artist']['top1'])
            self.history['artist_top5'].append(metrics['artist']['top5'])
            self.history['artist_f1'].append(metrics['artist']['f1'])

            self.history['genre_top1'].append(metrics['genre']['top1'])
            self.history['genre_top5'].append(metrics['genre']['top5'])
            self.history['genre_f1'].append(metrics['genre']['f1'])
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
            print(f"  - Style  | Top-1: {metrics['style']['top1']:>5.2f} % | Top-5: {metrics['style']['top5']:>5.2f} % | F1: {metrics['style']['f1']:.4f}")
            print(f"  - Artist | Top-1: {metrics['artist']['top1']:>5.2f} % | Top-5: {metrics['artist']['top5']:>5.2f} % | F1: {metrics['artist']['f1']:.4f}")
            print(f"  - Genre  | Top-1: {metrics['genre']['top1']:>5.2f} % | Top-5: {metrics['genre']['top5']:>5.2f} % | F1: {metrics['genre']['f1']:.4f}")
            print(f"  - Global F1: {metrics['global_f1']:.4f}\n")

            # save the model based on the new global F1
            if metrics['global_f1'] > self.best_f1:
                self.best_f1 = metrics['global_f1']
                torch.save(self.model.state_dict(), 'MultiHead_CRNN_classifier_BestModel.pth')
                print(f"*** New best model saved! (Global F1: {self.best_f1:.4f}) ***\n")

        return self.history