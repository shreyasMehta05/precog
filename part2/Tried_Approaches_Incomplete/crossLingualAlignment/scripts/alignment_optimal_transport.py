#!/usr/bin/env python3
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
import argparse

# Initialize Colorama for cross-platform colored output
init(autoreset=True)

def load_embeddings(path):
    print(Fore.YELLOW + f"Loading embeddings from {path}..." + Style.RESET_ALL)
    emb = KeyedVectors.load_word2vec_format(path)
    print(Fore.GREEN + f"Loaded {len(emb.index_to_key)} words from {path}." + Style.RESET_ALL)
    return emb

def load_dictionary(dict_path, src_emb, tgt_emb):
    print(Fore.YELLOW + f"Loading bilingual dictionary from {dict_path}..." + Style.RESET_ALL)
    word_pairs = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=Fore.CYAN + "Reading dictionary" + Style.RESET_ALL):
            src, tgt = line.strip().split('\t')
            if src in src_emb.key_to_index and tgt in tgt_emb.key_to_index:
                word_pairs.append((src, tgt))
    print(Fore.GREEN + f"Loaded {len(word_pairs)} bilingual pairs." + Style.RESET_ALL)
    return word_pairs

def build_matrices(word_pairs, src_emb, tgt_emb):
    src_words = [pair[0] for pair in word_pairs]
    tgt_words = [pair[1] for pair in word_pairs]
    X = np.vstack([src_emb[word] for word in tqdm(src_words, desc=Fore.BLUE + "Building source matrix" + Style.RESET_ALL)]).astype(np.float32)
    Y = np.vstack([tgt_emb[word] for word in tqdm(tgt_words, desc=Fore.BLUE + "Building target matrix" + Style.RESET_ALL)]).astype(np.float32)
    return X, Y

def optimal_transport_alignment(src_emb, tgt_emb, dictionary, reg=0.05, sample_size=5000, batch_size=1000):
    print(Fore.YELLOW + "\nStarting Optimal Transport alignment..." + Style.RESET_ALL)
    
    # Sample a subset of bilingual pairs
    pairs = dictionary
    if sample_size is not None and sample_size < len(pairs):
        indices = np.random.choice(len(pairs), sample_size, replace=False)
        pairs = [pairs[i] for i in indices]
        print(Fore.CYAN + f"Using a subset of {sample_size} bilingual pairs for OT." + Style.RESET_ALL)
    
    # Build matrices
    X, Y = build_matrices(pairs, src_emb, tgt_emb)
    
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
    P_normalized = n * P  # Now each row sums to 1
    
    # Compute barycenter of target embeddings for the sampled pairs
    Y_bar = np.dot(P_normalized, Y)
    
    # Compute the optimal transformation matrix using orthogonal Procrustes
    W, _ = orthogonal_procrustes(X, Y_bar)
    
    # Apply W to all source embeddings
    src_vectors = src_emb.vectors.astype(np.float32)
    aligned_vectors = np.dot(src_vectors, W)
    
    print(Fore.GREEN + "Optimal Transport alignment completed.\n" + Style.RESET_ALL)
    return aligned_vectors, W

def main():
    parser = argparse.ArgumentParser(description="Optimal Transport Alignment")
    parser.add_argument("--src_emb", type=str, required=True, help="Path to source (Hindi) embeddings")
    parser.add_argument("--tgt_emb", type=str, required=True, help="Path to target (English) embeddings")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to bilingual dictionary")
    parser.add_argument("--output_dir", type=str, default="ot_alignment_output", help="Directory to save outputs")
    parser.add_argument("--sample_size", type=int, default=5000, help="Sample size for bilingual pairs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for cost matrix computation")
    parser.add_argument("--reg", type=float, default=0.05, help="Regularization parameter")
    args = parser.parse_args()

    start = time.time()
    src_emb = load_embeddings(args.src_emb)
    tgt_emb = load_embeddings(args.tgt_emb)
    dictionary = load_dictionary(args.dict_path, src_emb, tgt_emb)
    aligned_vectors, W = optimal_transport_alignment(src_emb, tgt_emb, dictionary, reg=args.reg, sample_size=args.sample_size, batch_size=args.batch_size)
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "aligned_optimal_transport.npy"), aligned_vectors)
    np.save(os.path.join(args.output_dir, "optimal_transport_matrix.npy"), W)
    with open(os.path.join(args.output_dir, "vocabulary.txt"), 'w', encoding='utf-8') as f:
        for word in src_emb.index_to_key:
            f.write(word + "\n")
    end = time.time()
    print(Fore.CYAN + f"\nTotal time taken: {end - start:.2f} seconds" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
