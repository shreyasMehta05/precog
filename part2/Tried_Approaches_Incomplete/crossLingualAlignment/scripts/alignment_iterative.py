#!/usr/bin/env python3
import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
import os
from tqdm import tqdm
from colorama import Fore, Style, init
import time
import argparse

# Initialize Colorama for colored output
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
    X = np.vstack([src_emb[word] for word in tqdm(src_words, desc=Fore.BLUE + "Building source matrix" + Style.RESET_ALL)])
    Y = np.vstack([tgt_emb[word] for word in tqdm(tgt_words, desc=Fore.BLUE + "Building target matrix" + Style.RESET_ALL)])
    return X, Y

def iterative_self_learning(src_emb, tgt_emb, dictionary, max_iter=5, k=5, batch_size=1000, use_dict_target=True):
    print(Fore.YELLOW + "\nStarting Iterative Self-Learning alignment..." + Style.RESET_ALL)
    
    src_words = [pair[0] for pair in dictionary]
    tgt_words = [pair[1] for pair in dictionary]
    X, Y = build_matrices(dictionary, src_emb, tgt_emb)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    M = Y_t.transpose(0, 1).mm(X_t)
    U, S, V = torch.svd(M)
    W = V.mm(U.transpose(0, 1))
    W_np = W.cpu().numpy()
    
    src_vectors = src_emb.vectors
    tgt_vectors = Y if use_dict_target else tgt_emb.vectors
    
    for iteration in tqdm(range(max_iter), desc=Fore.YELLOW + "Iterative self-learning" + Style.RESET_ALL):
        aligned_vectors = np.dot(src_vectors, W_np)
        n_src = aligned_vectors.shape[0]
        nn_indices = []
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

def main():
    parser = argparse.ArgumentParser(description="Iterative Self-Learning Alignment")
    parser.add_argument("--src_emb", type=str, required=True, help="Path to source (Hindi) embeddings")
    parser.add_argument("--tgt_emb", type=str, required=True, help="Path to target (English) embeddings")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to bilingual dictionary")
    parser.add_argument("--output_dir", type=str, default="iterative_alignment_output", help="Directory to save outputs")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to consider")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for computing similarities")
    parser.add_argument("--use_dict_target", action="store_true", help="Use only dictionary target embeddings")
    args = parser.parse_args()
    
    start = time.time()
    src_emb = load_embeddings(args.src_emb)
    tgt_emb = load_embeddings(args.tgt_emb)
    dictionary = load_dictionary(args.dict_path, src_emb, tgt_emb)
    aligned_vectors, W = iterative_self_learning(src_emb, tgt_emb, dictionary, max_iter=args.max_iter, k=args.k, batch_size=args.batch_size, use_dict_target=args.use_dict_target)
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "aligned_iterative.npy"), aligned_vectors)
    np.save(os.path.join(args.output_dir, "iterative_matrix.npy"), W)
    with open(os.path.join(args.output_dir, "vocabulary.txt"), 'w', encoding='utf-8') as f:
        for word in src_emb.index_to_key:
            f.write(word + "\n")
    end = time.time()
    print(Fore.CYAN + f"\nTotal time taken: {end - start:.2f} seconds" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
