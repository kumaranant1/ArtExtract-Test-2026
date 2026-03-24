from random import random

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on all three tasks (artist, style, genre) and calculate the following metrics:
    - Top-1 Accuracy
    - Top-5 Accuracy
    - Macro F1 Score
    The function returns a dictionary containing the validation loss, global F1 score, and individual metrics for each task.
    """
    model.eval()
    running_val_loss = 0.0

    all_preds_a, all_targets_a = [], []
    all_preds_s, all_targets_s = [], []
    all_preds_g, all_targets_g = [], []

    with torch.no_grad():
        for images, artists, styles, genres in tqdm(val_loader, desc="[Validate]", leave=False):
            images = images.to(device)
            artists, styles, genres = artists.to(device), styles.to(device), genres.to(device)

            out_a, out_s, out_g = model(images)

            loss_a = criterion(out_a, artists)
            loss_s = criterion(out_s, styles)
            loss_g = criterion(out_g, genres)

            val_a = loss_a.item() if not torch.isnan(loss_a) else 0.0 # losses can be nan if in the batch all label is -1 for artist
            val_s = loss_s.item() if not torch.isnan(loss_s) else 0.0
            val_g = loss_g.item() if not torch.isnan(loss_g) else 0.0

            running_val_loss += (val_a + val_s + val_g)

            _, top5_a = out_a.topk(5, dim=1, largest=True, sorted=True)
            _, top5_s = out_s.topk(5, dim=1, largest=True, sorted=True)
            _, top5_g = out_g.topk(5, dim=1, largest=True, sorted=True)

            all_preds_a.append(top5_a.cpu())
            all_targets_a.append(artists.cpu())

            all_preds_s.append(top5_s.cpu())
            all_targets_s.append(styles.cpu())

            all_preds_g.append(top5_g.cpu())
            all_targets_g.append(genres.cpu())

    # concatenate the batched predictions into single tensors
    preds_a, targets_a = torch.cat(all_preds_a), torch.cat(all_targets_a)
    preds_s, targets_s = torch.cat(all_preds_s), torch.cat(all_targets_s)
    preds_g, targets_g = torch.cat(all_preds_g), torch.cat(all_targets_g)

    def calc_metrics(top5_preds, targets):
        """
        Calculates Top-1, Top-5 accuracies and Macro F1 score for a single task.
        """
        # mask out the -1 labels
        valid_mask = targets != -1
        valid_top5 = top5_preds[valid_mask]
        valid_targets = targets[valid_mask]

        if len(valid_targets) == 0:
            return 0.0, 0.0, 0.0

        # extract Top-1 Predictions
        top1_preds = valid_top5[:, 0]

        # calculate accuracies
        correct_top1 = top1_preds.eq(valid_targets).sum().item()

        # check if the true target exists anywhere inside the top 5 predictions
        correct_top5 = valid_top5.eq(valid_targets.view(-1, 1).expand_as(valid_top5)).sum().item()

        top1_acc = (correct_top1 / len(valid_targets)) * 100
        top5_acc = (correct_top5 / len(valid_targets)) * 100

        # calculate macro F1
        f1 = f1_score(valid_targets.numpy(), top1_preds.numpy(), average='macro', zero_division=0)

        return top1_acc, top5_acc, f1

    # calculate metrics for all three tasks using the new helper
    top1_a, top5_a, f1_a = calc_metrics(preds_a, targets_a)
    top1_s, top5_s, f1_s = calc_metrics(preds_s, targets_s)
    top1_g, top5_g, f1_g = calc_metrics(preds_g, targets_g)

    val_loss = running_val_loss / len(val_loader)
    global_f1 = (f1_a + f1_s + f1_g) / 3.0

    metrics = {
        'val_loss': val_loss,
        'global_f1': global_f1,
        'artist': {'top1': top1_a, 'top5': top5_a, 'f1': f1_a},
        'style': {'top1': top1_s, 'top5': top5_s, 'f1': f1_s},
        'genre': {'top1': top1_g, 'top5': top5_g, 'f1': f1_g}
    }

    return metrics


def evaluate_retrieval(index, embeddings, sample_size=1000, top_k=5):
    
    num_total = len(embeddings)
    query_indices = random.sample(range(num_total), min(sample_size, num_total))
    
    total_cos_sim = 0.0

    for idx in query_indices:
        query_vector = embeddings[idx].reshape(1, -1)

        # search Top-K (+1 to ignore the 100% match to itself)
        cos_scores, matched_indices = index.search(query_vector, top_k + 1)
        
        # calculate average cosine confidence
        avg_query_cos = np.mean(cos_scores[0][1:])
        total_cos_sim += avg_query_cos

    final_avg_cos = (total_cos_sim / len(query_indices)) * 100
    
    return final_avg_cos