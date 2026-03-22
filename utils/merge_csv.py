import pandas as pd
import os
from pathlib import Path

def MERGE_CSV(csv_dir:str ="./data/wikiart_csv", split_name:str ='train', output_dir:str ="./data"):
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    # Load CSVs
    df_artist = pd.read_csv(csv_dir / f'artist_{split_name}.csv', header=None, names=['filepath', 'artist_label'])
    df_style  = pd.read_csv(csv_dir / f'style_{split_name}.csv', header=None, names=['filepath', 'style_label'])
    df_genre  = pd.read_csv(csv_dir / f'genre_{split_name}.csv', header=None, names=['filepath', 'genre_label'])

    # Use outer join to keep everything
    master_df = df_artist.merge(df_style, on='filepath', how='outer')
    master_df = master_df.merge(df_genre, on='filepath', how='outer')

    # fill missing labels with -1 and convert back to integers
    master_df.fillna(-1, inplace=True)
    master_df['artist_label'] = master_df['artist_label'].astype(int)
    master_df['style_label'] = master_df['style_label'].astype(int)
    master_df['genre_label'] = master_df['genre_label'].astype(int)

    # save
    output_path = output_dir / f'{split_name}_metadata.csv'
    master_df.to_csv(output_path, index=False)
    
    print(f"Dataset size with Outer Join: {len(master_df)} images!")
    return master_df

if __name__ == "__main__":
    print("--- Starting Metadata Consolidation ---")
    MERGE_CSV(split_name='train')
    MERGE_CSV(split_name='val')
    print("--- Metadata Consolidation Complete ---")