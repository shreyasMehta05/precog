#!/usr/bin/env python
import argparse
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embedding_file, max_vocab=None, device=None):
    """
    Loads embeddings from a text file.
    Skips header if the first line contains two numeric tokens.
    Uses a progress bar and returns a dictionary mapping word -> tensor.
    """
    embeddings = {}
    # Try to count total lines for progress bar (if possible)
    total = None
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    with open(embedding_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        header_skipped = False
        # If first line is header, skip it.
        if len(first_line) == 2:
            try:
                int(first_line[0])
                int(first_line[1])
                header_skipped = True
            except ValueError:
                header_skipped = False
        if not header_skipped:
            if len(first_line) >= 2:
                word = first_line[0]
                vec = torch.tensor([float(x) for x in first_line[1:]], dtype=torch.float32)
                if device is not None:
                    vec = vec.to(device)
                embeddings[word] = vec

        adjusted_total = (total - 1) if header_skipped and total is not None else total
        for line in tqdm(f, total=adjusted_total, desc=f"Loading {os.path.basename(embedding_file)}"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            if device is not None:
                vec = vec.to(device)
            embeddings[word] = vec
            if max_vocab is not None and len(embeddings) >= max_vocab:
                break

    if embeddings:
        dim = next(iter(embeddings.values())).shape[0]
    else:
        dim = 0
    return embeddings, dim

def load_bilingual_dictionary(dict_file):
    """
    Loads bilingual dictionary from a file.
    Each line: hindi_word english_word
    Returns a list of (hindi_word, english_word) pairs.
    """
    dictionary = []
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading bilingual dictionary"):
            parts = line.strip().split()
            if len(parts) >= 2:
                dictionary.append((parts[0], parts[1]))
    return dictionary

def get_alignment_matrices(bilingual_dict, hindi_embeddings, english_embeddings, device=None):
    """
    For each dictionary pair (hindi, english) available in both embeddings,
    build matrices X (Hindi) and Y (English).
    """
    X_list, Y_list, valid_pairs = [], [], []
    for hindi_word, english_word in tqdm(bilingual_dict, desc="Building alignment matrices", total=len(bilingual_dict)):
        if hindi_word in hindi_embeddings and english_word in english_embeddings:
            X_list.append(hindi_embeddings[hindi_word])
            Y_list.append(english_embeddings[english_word])
            valid_pairs.append((hindi_word, english_word))
    X = torch.stack(X_list) if X_list else torch.tensor([])
    Y = torch.stack(Y_list) if Y_list else torch.tensor([])
    return X, Y, valid_pairs

def procrustes(X, Y):
    """
    Computes the optimal orthogonal transformation R such that X @ R ≈ Y.
    Saves the matrices for later inspection.
    """
    A = torch.matmul(X.T, Y)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(Vh.T, U.T)
    return R

def apply_transformation_chunked(embeddings, R, chunk_size=10000):
    """
    Applies transformation R to embeddings in chunks to avoid VRAM issues.
    Returns a new dictionary of transformed embeddings.
    """
    transformed = {}
    items = list(embeddings.items())
    num_items = len(items)
    for i in tqdm(range(0, num_items, chunk_size), desc="Applying transformation in chunks"):
        chunk = items[i: i + chunk_size]
        words = [item[0] for item in chunk]
        vecs = [item[1] for item in chunk]
        vecs = torch.stack(vecs)  # (batch_size, d)
        vecs = torch.matmul(vecs, R)
        for j, word in enumerate(words):
            transformed[word] = vecs[j]
    return transformed

def save_embeddings_text(embeddings, output_file):
    """
    Saves embeddings to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, vec in tqdm(embeddings.items(), desc="Saving embeddings as text"):
            vec_cpu = vec.detach().cpu().numpy()
            vec_str = ' '.join(map(str, vec_cpu.tolist()))
            f.write(f"{word} {vec_str}\n")

def save_tensor(tensor, filename):
    """
    Saves a torch tensor to disk.
    """
    torch.save(tensor, filename)

def visualize_alignment(english_embeddings, aligned_hindi_embeddings, bilingual_dict, sample_size=200, output_file='visualizations/tsne_alignment.png'):
    """
    Creates a t-SNE plot of a sample of aligned Hindi and corresponding English embeddings.
    """
    sample_pairs = random.sample(bilingual_dict, min(sample_size, len(bilingual_dict)))
    hindi_vecs, english_vecs, labels = [], [], []
    for hindi_word, english_word in sample_pairs:
        if hindi_word in aligned_hindi_embeddings and english_word in english_embeddings:
            hindi_vecs.append(aligned_hindi_embeddings[hindi_word].detach().cpu().numpy())
            english_vecs.append(english_embeddings[english_word].detach().cpu().numpy())
            labels.append(f"{hindi_word}/{english_word}")
    import numpy as np
    if len(hindi_vecs) == 0:
        print("Not enough samples for visualization.")
        return
    data = np.concatenate([np.array(hindi_vecs), np.array(english_vecs)], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:len(hindi_vecs), 0], tsne_result[:len(hindi_vecs), 1], c='red', label='Aligned Hindi')
    plt.scatter(tsne_result[len(hindi_vecs):, 0], tsne_result[len(hindi_vecs):, 1], c='blue', label='English')
    # Optionally annotate a few points.
    for i, label in enumerate(labels):
        if i % (max(1, len(labels)//10)) == 0:
            plt.annotate(label, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
    plt.legend()
    plt.title("t-SNE Visualization of Aligned Hindi vs. English Embeddings")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved t-SNE alignment visualization to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cross-lingual alignment using Procrustes analysis with GPU acceleration, chunk processing, and visualization."
    )
    parser.add_argument('--english', type=str, required=True, help="Path to English embeddings file")
    parser.add_argument('--hindi', type=str, required=True, help="Path to Hindi embeddings file")
    parser.add_argument('--dict', type=str, required=True, help="Path to bilingual dictionary file (hindi english per line)")
    parser.add_argument('--output', type=str, required=True, help="Output text file for aligned Hindi embeddings")
    parser.add_argument('--max_vocab', type=int, default=None, help="Maximum number of words to load from embeddings")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--chunk_size', type=int, default=10000, help="Chunk size for transformation to avoid VRAM issues")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading English embeddings...")
    english_embeddings, en_dim = load_embeddings(args.english, max_vocab=args.max_vocab, device=device)
    print(f"Loaded {len(english_embeddings)} English embeddings with dimension {en_dim}.")

    print("Loading Hindi embeddings...")
    hindi_embeddings, hi_dim = load_embeddings(args.hindi, max_vocab=args.max_vocab, device=device)
    print(f"Loaded {len(hindi_embeddings)} Hindi embeddings with dimension {hi_dim}.")

    if en_dim != hi_dim:
        raise ValueError(f"Dimension mismatch: English dim = {en_dim}, Hindi dim = {hi_dim}")

    print("Loading bilingual dictionary...")
    bilingual_dict = load_bilingual_dictionary(args.dict)
    print(f"Loaded {len(bilingual_dict)} dictionary entries.")

    print("Building alignment matrices...")
    X, Y, valid_pairs = get_alignment_matrices(bilingual_dict, hindi_embeddings, english_embeddings, device=device)
    print(f"Using {X.shape[0]} valid dictionary pairs for alignment.")

    # Save intermediate alignment matrices.
    os.makedirs("data", exist_ok=True)
    save_tensor(X, "data/X_tensor.pt")
    save_tensor(Y, "data/Y_tensor.pt")
    print("Saved alignment matrices X and Y.")

    print("Computing transformation matrix using Procrustes analysis...")
    R = procrustes(X, Y)
    print("Transformation matrix computed.")
    save_tensor(R, "data/transformation_R.pt")
    print("Saved transformation matrix R.")

    print("Applying transformation to Hindi embeddings in chunks...")
    aligned_hindi_embeddings = apply_transformation_chunked(hindi_embeddings, R, chunk_size=args.chunk_size)
    print("Transformation applied.")

    # Save aligned Hindi embeddings as a Torch file.
    save_tensor(aligned_hindi_embeddings, "data/aligned_hindi_embeddings.pt")
    print("Saved aligned Hindi embeddings as a Torch file.")

    print("Saving aligned Hindi embeddings as text...")
    save_embeddings_text(aligned_hindi_embeddings, args.output)
    print("Aligned Hindi embeddings saved as text.")

    print("Generating t-SNE visualization of aligned embeddings...")
    visualize_alignment(english_embeddings, aligned_hindi_embeddings, bilingual_dict, output_file="visualizations/tsne_alignment.png")

    print("Alignment complete!")
