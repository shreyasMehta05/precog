"""
download_data.py
Handles the downloading and preparation of evaluation datasets from various sources.
"""

import os
import requests
import gdown
import zipfile
from tqdm import tqdm
from pathlib import Path

class DatasetDownloader:
    def __init__(self):
        self.base_directory = Path("data")
        self.word_sets_directory = self.base_directory / "word_sets"
        self.models_directory = self.base_directory / "models"
        
        # Create necessary directories if they don't exist
        self.word_sets_directory.mkdir(parents=True, exist_ok=True)
        self.models_directory.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, destination_path: Path, description: str = "Downloading"):
        """Download a file from a URL and display a progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(destination_path, 'wb') as file, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                file_size = file.write(chunk)
                progress_bar.update(file_size)

    def download_word_sets(self):
        """Download various word sets used for bias evaluation."""
        
        # Download WEAT word sets from Caliskan et al.
        weat_url = "https://raw.githubusercontent.com/w4ngatang/sent-bias/master/data/weat.json"
        self.download_file(weat_url, self.word_sets_directory / "weat.json", "Downloading WEAT word sets")

        # Download professional word sets from Bolukbasi et al.
        professions_url = "https://raw.githubusercontent.com/tolga-b/debiaswe/master/data/professions.json"
        self.download_file(professions_url, self.word_sets_directory / "professions.json", "Downloading profession word sets")

    def download_word_embeddings(self):
        """Download pre-trained word embeddings."""
        
        # Download Google’s word2vec embeddings
        word2vec_url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        word2vec_path = self.models_directory / "GoogleNews-vectors-negative300.bin.gz"
        
        if not word2vec_path.exists():
            print("Downloading word2vec embeddings (this may take some time)...")
            gdown.download(word2vec_url, str(word2vec_path), quiet=False)
        
        # For GloVe embeddings, users must download manually due to licensing restrictions
        print("\nNote: Please manually download GloVe embeddings from:")
        print("https://nlp.stanford.edu/data/glove.840B.300d.zip")

def main():
    downloader = DatasetDownloader()
    
    print("Starting data download process...")
    
    # Download word sets
    print("\nDownloading word sets...")
    downloader.download_word_sets()
    
    # Download word embeddings
    print("\nDownloading word embeddings...")
    downloader.download_word_embeddings()
    
    print("\nDownload complete! Directory structure:")
    print("\ndata/")
    print("├── word_sets/")
    print("│   ├── weat.json")
    print("│   └── professions.json")
    print("└── models/")
    print("    └── GoogleNews-vectors-negative300.bin.gz")

if __name__ == "__main__":
    main()
