import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from gensim.models import KeyedVectors
import os
from tqdm import tqdm
from colorama import Fore, Style, init
import time

# Initialize Colorama for cross-platform colored output
init(autoreset=True)

class CrossLingualAligner:
    def __init__(self, src_emb_path, tgt_emb_path, dict_path):
        """
        Initialize the aligner with paths to embeddings and dictionary.
        
        Args:
            src_emb_path: Path to source (Hindi) embeddings (FastText .vec format)
            tgt_emb_path: Path to target (English) embeddings (FastText .vec format)
            dict_path: Path to training bilingual dictionary (tab-separated)
        """
        print(Fore.YELLOW + "Loading source embeddings..." + Style.RESET_ALL)
        self.src_emb = KeyedVectors.load_word2vec_format(src_emb_path)
        print(Fore.GREEN + f"Loaded {len(self.src_emb.index_to_key)} source words." + Style.RESET_ALL)
        
        print(Fore.YELLOW + "Loading target embeddings..." + Style.RESET_ALL)
        self.tgt_emb = KeyedVectors.load_word2vec_format(tgt_emb_path)
        print(Fore.GREEN + f"Loaded {len(self.tgt_emb.index_to_key)} target words." + Style.RESET_ALL)
        
        print(Fore.YELLOW + "Loading bilingual dictionary..." + Style.RESET_ALL)
        self.train_dict = self._load_dictionary(dict_path)
        print(Fore.GREEN + f"Loaded {len(self.train_dict)} bilingual pairs." + Style.RESET_ALL)
        
    def _load_dictionary(self, dict_path):
        """Load bilingual dictionary from file."""
        word_pairs = []
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=Fore.CYAN + "Reading dictionary" + Style.RESET_ALL):
                src, tgt = line.strip().split('\t')
                # Use key_to_index to check membership in Gensim 4.x
                if src in self.src_emb.key_to_index and tgt in self.tgt_emb.key_to_index:
                    word_pairs.append((src, tgt))
        return word_pairs
    
    def _get_aligned_matrices(self):
        """Extract matrices X (source/Hindi) and Y (target/English) using the training dictionary."""
        src_words = [pair[0] for pair in self.train_dict]
        tgt_words = [pair[1] for pair in self.train_dict]
        
        X_list = []
        for word in tqdm(src_words, desc=Fore.BLUE + "Building source matrix" + Style.RESET_ALL):
            X_list.append(self.src_emb[word])
        X = np.vstack(X_list)
        
        Y_list = []
        for word in tqdm(tgt_words, desc=Fore.BLUE + "Building target matrix" + Style.RESET_ALL):
            Y_list.append(self.tgt_emb[word])
        Y = np.vstack(Y_list)
        return X, Y
    
    def align_procrustes(self):
        """
        Align embeddings using Procrustes Analysis with GPU acceleration.
        Returns:
            aligned_vectors: Aligned source embeddings (numpy array)
            W: Transformation matrix (numpy array)
        """
        print(Fore.YELLOW + "\nStarting Procrustes alignment..." + Style.RESET_ALL)
        X, Y = self._get_aligned_matrices()
        
        # Set device to GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(Fore.MAGENTA + f"Using device: {device}" + Style.RESET_ALL)
        
        # Convert training matrices to torch tensors and move to device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
        
        # Compute cross-covariance matrix: M = Y^T X
        M = Y_t.transpose(0, 1).mm(X_t)
        # Compute SVD of M on GPU
        U, S, V = torch.svd(M)
        # Optimal orthogonal transformation: W = V * U^T
        W = V.mm(U.transpose(0, 1))
        
        # Transform all source embeddings using W
        src_vectors = self.src_emb.vectors  # (n, d) numpy array
        src_t = torch.tensor(src_vectors, dtype=torch.float32, device=device)
        aligned_t = src_t.mm(W)
        aligned_vectors = aligned_t.cpu().numpy()
        
        print(Fore.GREEN + "Procrustes alignment completed.\n" + Style.RESET_ALL)
        return aligned_vectors, W.cpu().numpy()
    
    def align_cca(self, n_components=None):
        """
        Align embeddings using Canonical Correlation Analysis (CCA).
        This method uses scikit-learn's implementation (CPU-based).
        
        Args:
            n_components: Number of CCA components (default: minimum dimension)
        
        Returns:
            aligned_vectors: Transformed source embeddings via CCA (numpy array)
            cca: Fitted CCA model
        """
        print(Fore.YELLOW + "Starting CCA alignment..." + Style.RESET_ALL)
        X, Y = self._get_aligned_matrices()
        
        if n_components is None:
            n_components = min(X.shape[1], Y.shape[1])
        
        # Increase max_iter to help convergence
        cca = CCA(n_components=n_components, max_iter=2000)
        cca.fit(X, Y)
        
        src_vectors = self.src_emb.vectors
        aligned_vectors = cca.transform(src_vectors)[0]
        
        print(Fore.GREEN + "CCA alignment completed.\n" + Style.RESET_ALL)
        return aligned_vectors, cca
    
    def save_alignments(self, output_dir):
        """Save aligned embeddings and transformation matrices to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Procrustes alignment (GPU accelerated)
        proc_vectors, proc_matrix = self.align_procrustes()
        np.save(os.path.join(output_dir, 'aligned_procrustes.npy'), proc_vectors)
        np.save(os.path.join(output_dir, 'procrustes_matrix.npy'), proc_matrix)
        
        # CCA alignment (CPU-based)
        cca_vectors, cca_model = self.align_cca()
        np.save(os.path.join(output_dir, 'aligned_cca.npy'), cca_vectors)
        
        # Save source vocabulary for reference
        with open(os.path.join(output_dir, 'vocabulary.txt'), 'w', encoding='utf-8') as f:
            for word in self.src_emb.index_to_key:
                f.write(f"{word}\n")
        print(Fore.GREEN + f"\nAll alignments saved to {output_dir}" + Style.RESET_ALL)

if __name__ == "__main__":
    # Example usage:
    start_time = time.time()
    aligner = CrossLingualAligner(
        src_emb_path="data/hindi_embeddings.vec",
        tgt_emb_path="data/english_embeddings.vec",
        dict_path="data/hi_en_dictionary.txt"
    )
    end_time = time.time()
    aligner.save_alignments("output_alignments")
    print(Fore.CYAN + f"\nTotal time taken: {end_time - start_time:.2f} seconds" + Style.RESET_ALL)
