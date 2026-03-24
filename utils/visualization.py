import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def load_class_names(filepath):
    """
    Reads the class txt file and create dict for id: name 
    """
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def plot_outliers(model, val_loader, val_dataset, device, artist_names, style_names, genre_names):
    """
    Plots the top 10 outliers for each task (artist, style, genre) based on the model's confidence in its incorrect predictions.
    Args:
        model: The trained CRNN model.
        val_loader: DataLoader for the validation set.
        val_dataset: The validation dataset (used to fetch original image paths).
        device: device (CPU/GPU) to run the model on.
        artist_names: Dictionary mapping artist class indices to names.
        style_names: Dictionary mapping style class indices to names.
        genre_names: Dictionary mapping genre class indices to names.
    """
    model.eval()

    outliers_artist = []
    outliers_style = []
    outliers_genre = []

    current_idx = 0

    with torch.no_grad():
        for images, artists, styles, genres in tqdm(val_loader, desc="Processing Validation Set"):
            images = images.to(device)
            
            # get raw logits
            out_a, out_s, out_g = model(images)
            
            # apply Softmax to convert logits to probs
            probs_a = F.softmax(out_a, dim=1)
            probs_s = F.softmax(out_s, dim=1)
            probs_g = F.softmax(out_g, dim=1)
            
            # get the highest probability and the predicted class index
            conf_a, pred_a = torch.max(probs_a, dim=1)
            conf_s, pred_s = torch.max(probs_s, dim=1)
            conf_g, pred_g = torch.max(probs_g, dim=1)
            
            artists, styles, genres = artists.cpu().numpy(), styles.cpu().numpy(), genres.cpu().numpy()
            pred_a, pred_s, pred_g = pred_a.cpu().numpy(), pred_s.cpu().numpy(), pred_g.cpu().numpy()
            conf_a, conf_s, conf_g = conf_a.cpu().numpy(), conf_s.cpu().numpy(), conf_g.cpu().numpy()
            
            for i in range(len(images)):
                # artist outliers
                true_a = artists[i]
                if true_a != -1 and true_a != pred_a[i]:
                    outliers_artist.append({
                        'idx': current_idx + i, 
                        'true': artist_names.get(true_a, f"Unknown ({true_a})"), 
                        'pred': artist_names.get(pred_a[i], f"Unknown ({pred_a[i]})"), 
                        'conf': conf_a[i] * 100
                    })
                    
                # style outliers
                true_s = styles[i]
                if true_s != -1 and true_s != pred_s[i]:
                    outliers_style.append({
                        'idx': current_idx + i, 
                        'true': style_names.get(true_s, f"Unknown ({true_s})"), 
                        'pred': style_names.get(pred_s[i], f"Unknown ({pred_s[i]})"), 
                        'conf': conf_s[i] * 100
                    })
                    
                # genre outliers
                true_g = genres[i]
                if true_g != -1 and true_g != pred_g[i]:
                    outliers_genre.append({
                        'idx': current_idx + i, 
                        'true': genre_names.get(true_g, f"Unknown ({true_g})"), 
                        'pred': genre_names.get(pred_g[i], f"Unknown ({pred_g[i]})"), 
                        'conf': conf_g[i] * 100
                    })
                    
            current_idx += len(images)
    
    outliers_artist.sort(key=lambda x: x['conf'], reverse=True)
    outliers_style.sort(key=lambda x: x['conf'], reverse=True)
    outliers_genre.sort(key=lambda x: x['conf'], reverse=True)

    def plot_top_10_outliers(outliers_list, dataset, task_name):
        
        print(f"\n--- Plotting Top 10 Outliers for {task_name.upper()} ---")
        top_10 = outliers_list[:10]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        axes = axes.flatten()
        
        for i, outlier in enumerate(top_10):
            # We fetch the original image from the hard drive so it looks beautiful
            img_path = os.path.join(dataset.root_dir, dataset.metadata.iloc[outlier['idx']]['filepath'])
            raw_image = Image.open(img_path).convert('RGB')
            
            ax = axes[i]
            ax.imshow(raw_image)
            ax.axis('off')
            
            title_text = f"True: {outlier['true']}\nPred: {outlier['pred']}\nConf: {outlier['conf']:.1f}%"
            
            # color the text red to indicate an error
            ax.set_title(title_text, color='darkred', fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        plt.show()

    # Call the plotting helper for all three lists
    plot_top_10_outliers(outliers_artist, val_dataset, "Artist")
    plot_top_10_outliers(outliers_style, val_dataset, "Style")
    plot_top_10_outliers(outliers_genre, val_dataset, "Genre")

