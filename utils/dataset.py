import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class NGADataset(Dataset):
    """
    Dataset for loading NGA images from provided input directory.
    """
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, img_name


class WikiArtDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Args:
            csv_file: Path to the csv file with annotations.
            root_dir: Directory with all the images.
            transform : transform to be applied on a sample.
        """

        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx]['filepath'])

        try:
            # load the image
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        # extract labels for artist, style, and genre
        artist_label = int(self.metadata.iloc[idx]['artist_label'])
        style_label = int(self.metadata.iloc[idx]['style_label'])
        genre_label = int(self.metadata.iloc[idx]['genre_label'])

        # return the image, artist, style and genre
        return (image,
                torch.tensor(artist_label, dtype=torch.long),
                torch.tensor(style_label, dtype=torch.long),
                torch.tensor(genre_label, dtype=torch.long))

