import numpy as np
import torch
from gensim.models import KeyedVectors
import ot
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
import os
from tqdm import tqdm
from colorama import Fore, Style, init
import time

# Initialize Colorama for cross-platform colored output
init(autoreset=True)

class AdvancedCrossLingualAligner:
    def __init__(self, src_emb_path, tgt_emb_path, dict_path):
        """
        Initialize the aligner with paths to embeddings and dictionary.
        
        Args:
            src_emb_path: Path to source (Hindi) embeddings (FastText .vec format)
            tgt_emb_path: Path to target (English) embeddings (FastText .vec format)
            dict_path: Path to training dictionary (tab-separated)
        """
        print(Fore.YELLOW + "Loading source embeddings..." + Style.RESET_ALL)
        # Assuming the Hindi embeddings file includes a header.
        self.src_emb = KeyedVectors.load_word2vec_format(src_emb_path)
        print(Fore.GREEN + f"Loaded {len(self.src_emb.index_to_key)} source words." + Style.RESET_ALL)
        
        print(Fore.YELLOW + "Loading target embeddings..." + Style.RESET_ALL)
        # For the filtered English embeddings, if there is no header, pass no_header=True.
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
                # Use key_to_index for Gensim 4.x compatibility
                if src in self.src_emb.key_to_index and tgt in self.tgt_emb.key_to_index:
                    word_pairs.append((src, tgt))
        return word_pairs

    def align_optimal_transport(self, reg=0.05, sample_size=5000, batch_size=1000):
        """
        Memory-efficient alignment using Optimal Transport with entropic regularization.
        
        Args:
            reg: Entropic regularization parameter.
            sample_size: Number of bilingual pairs to use (reduced to save memory).
            batch_size: Size of batches for computing the cost matrix.
        
        Returns:
            aligned_vectors: Transformed source embeddings (applied to all source embeddings).
            W: Transformation matrix (of shape (d,d)).
        """
        print(Fore.YELLOW + "\nStarting Optimal Transport alignment..." + Style.RESET_ALL)
        
        # Sample a subset of bilingual pairs
        pairs = self.train_dict
        if sample_size is not None and sample_size < len(pairs):
            indices = np.random.choice(len(pairs), sample_size, replace=False)
            pairs = [pairs[i] for i in indices]
            print(Fore.CYAN + f"Using a subset of {sample_size} bilingual pairs for OT." + Style.RESET_ALL)
        
        src_words = [pair[0] for pair in pairs]
        tgt_words = [pair[1] for pair in pairs]
        
        # Build source and target matrices using float32 for lower memory usage
        X = np.vstack([self.src_emb[word] for word in tqdm(src_words, desc=Fore.BLUE + "Building source matrix" + Style.RESET_ALL)]).astype(np.float32)
        Y = np.vstack([self.tgt_emb[word] for word in tqdm(tgt_words, desc=Fore.BLUE + "Building target matrix" + Style.RESET_ALL)]).astype(np.float32)
        
        print(Fore.YELLOW + "Computing cost matrix in batches..." + Style.RESET_ALL)
        n = X.shape[0]
        C = np.zeros((n, n), dtype=np.float32)
        for i in tqdm(range(0, n, batch_size), desc=Fore.BLUE + "Processing batches" + Style.RESET_ALL):
            end = min(i + batch_size, n)
            batch_sim = cosine_similarity(X[i:end], Y)
            C[i:end] = -batch_sim  # Negative cosine similarity
        
        a = np.ones(n, dtype=np.float32) / n
        b = np.ones(n, dtype=np.float32) / n
        
        print(Fore.YELLOW + "Computing optimal transport plan..." + Style.RESET_ALL)
        P = ot.sinkhorn(a, b, C, reg, numItermax=1000)
        # Each row of P sums to 1/n. Multiply by n to normalize.
        P_normalized = n * P
        
        # Compute barycenter of target embeddings for the sampled pairs:
        Y_bar = np.dot(P_normalized, Y)
        
        # Compute the optimal transformation matrix W using orthogonal Procrustes:
        W, _ = orthogonal_procrustes(X, Y_bar)  # W shape: (d, d)
        
        # Apply W to all source embeddings
        src_vectors = self.src_emb.vectors.astype(np.float32)  # shape (N, d)
        aligned_vectors = np.dot(src_vectors, W)  # shape (N, d)
        
        print(Fore.GREEN + "Optimal Transport alignment completed.\n" + Style.RESET_ALL)
        return aligned_vectors, W

    def align_iterative_learning(self, max_iter=5, k=5, batch_size=1000, use_dict_target=True):
        """
        Memory-efficient iterative self-learning alignment.
        
        Args:
            max_iter: Maximum number of iterations.
            k: Number of nearest neighbors to consider.
            batch_size: Batch size for computing similarities.
            use_dict_target: If True, use only the target embeddings from the bilingual dictionary (to save memory).
        
        Returns:
            final_aligned_vectors: Aligned source embeddings.
            W: Final transformation matrix.
        """
        print(Fore.YELLOW + "\nStarting Iterative Self-Learning alignment..." + Style.RESET_ALL)
        
        # Use dictionary pairs for iterative learning to reduce memory usage
        src_words = [pair[0] for pair in self.train_dict]
        tgt_words = [pair[1] for pair in self.train_dict]
        
        X = np.vstack([self.src_emb[word] for word in tqdm(src_words, desc=Fore.BLUE + "Building source matrix" + Style.RESET_ALL)])
        Y = np.vstack([self.tgt_emb[word] for word in tqdm(tgt_words, desc=Fore.BLUE + "Building target matrix" + Style.RESET_ALL)])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
        M = Y_t.transpose(0, 1).mm(X_t)
        U, S, V = torch.svd(M)
        W = V.mm(U.transpose(0, 1))
        W_np = W.cpu().numpy()
        
        src_vectors = self.src_emb.vectors
        if use_dict_target:
            tgt_vectors = Y  # Use target embeddings from dictionary
        else:
            tgt_vectors = self.tgt_emb.vectors  # Use full target vocabulary
        
        for iteration in tqdm(range(max_iter), desc=Fore.YELLOW + "Iterative self-learning" + Style.RESET_ALL):
            # Transform source embeddings
            aligned_vectors = np.dot(src_vectors, W_np)
            n_src = aligned_vectors.shape[0]
            nn_indices = []
            # Compute cosine similarities in batches
            for i in tqdm(range(0, n_src, batch_size), desc=Fore.BLUE + f"Computing similarities (iter {iteration+1})" + Style.RESET_ALL):
                end = min(i + batch_size, n_src)
                batch_sim = cosine_similarity(aligned_vectors[i:end], tgt_vectors)
                batch_nn = np.argsort(-batch_sim, axis=1)[:, :k]
                nn_indices.extend(batch_nn)
            nn_indices = np.vstack(nn_indices)
            
            new_X = []
            new_Y = []
            for i, nns in enumerate(nn_indices):
                new_X.extend([src_vectors[i]] * k)
                new_Y.extend([tgt_vectors[j] for j in nns])
            new_X = np.vstack(new_X)
            new_Y = np.vstack(new_Y)
            
            new_X_t = torch.tensor(new_X, dtype=torch.float32, device=device)
            new_Y_t = torch.tensor(new_Y, dtype=torch.float32, device=device)
            M_new = new_Y_t.transpose(0, 1).mm(new_X_t)
            U_new, S_new, V_new = torch.svd(M_new)
            W = V_new.mm(U_new.transpose(0, 1))
            W_np = W.cpu().numpy()
        
        final_aligned_vectors = np.dot(src_vectors, W_np)
        print(Fore.GREEN + "Iterative Self-Learning alignment completed.\n" + Style.RESET_ALL)
        return final_aligned_vectors, W_np

    def save_alignments(self, output_dir):
        """Save aligned embeddings and transformation matrices to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimal Transport alignment
        ot_vectors, ot_matrix = self.align_optimal_transport(sample_size=5000, batch_size=1000)
        np.save(os.path.join(output_dir, 'aligned_optimal_transport.npy'), ot_vectors)
        np.save(os.path.join(output_dir, 'optimal_transport_matrix.npy'), ot_matrix)
        
        # Iterative Self-Learning alignment
        isl_vectors, isl_matrix = self.align_iterative_learning(max_iter=5, k=5, batch_size=1000, use_dict_target=True)
        np.save(os.path.join(output_dir, 'aligned_iterative.npy'), isl_vectors)
        np.save(os.path.join(output_dir, 'iterative_matrix.npy'), isl_matrix)
        
        # Save source vocabulary for reference
        with open(os.path.join(output_dir, 'vocabulary.txt'), 'w', encoding='utf-8') as f:
            for word in self.src_emb.index_to_key:
                f.write(f"{word}\n")
        print(Fore.GREEN + f"\nAll alignments saved to {output_dir}" + Style.RESET_ALL)

if __name__ == "__main__":
    start_time = time.time()
    print(Fore.CYAN + "Initializing Advanced Cross-Lingual Aligner..." + Style.RESET_ALL)
    aligner = AdvancedCrossLingualAligner(
        src_emb_path="data/hindi_embeddings.vec",
        tgt_emb_path="data/english_filtered.vec",
        dict_path="data/hi_en_dictionary.txt"
    )
    aligner.save_alignments("output_alignments")
    end_time = time.time()
    print(Fore.CYAN + f"\nTotal time taken: {end_time - start_time:.2f} seconds" + Style.RESET_ALL)
