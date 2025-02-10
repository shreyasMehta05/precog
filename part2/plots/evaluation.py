#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import numpy as np

def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embedding_file, max_vocab=None, device=None):
    """
    Loads embeddings from a file.
    If the file ends with '.pt', it's loaded as a Torch binary file.
    Otherwise, it's assumed to be a text file.
    """
    if embedding_file.endswith('.pt'):
        embeddings = torch.load(embedding_file, map_location=device)
        return embeddings

    embeddings = {}
    total = None
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    with open(embedding_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        header_skipped = False
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
    return embeddings

def load_bilingual_dictionary(dict_file):
    """
    Loads a bilingual dictionary from file.
    Each line should contain: hindi_word english_word
    Returns a list of (hindi_word, english_word) tuples.
    """
    dictionary = []
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading bilingual dictionary"):
            parts = line.strip().split()
            if len(parts) >= 2:
                dictionary.append((parts[0], parts[1]))
    return dictionary

def build_embedding_matrix(embeddings, word_list):
    """
    Given a list of words, builds a tensor matrix of embeddings.
    Returns the matrix and a list of valid words.
    """
    matrix = []
    valid_words = []
    for word in tqdm(word_list, desc="Building embedding matrix", total=len(word_list)):
        if word in embeddings:
            matrix.append(embeddings[word])
            valid_words.append(word)
    if matrix:
        matrix = torch.stack(matrix)
    else:
        matrix = torch.tensor([])
    return matrix, valid_words

def evaluate_alignment(aligned_hindi_embeddings, english_embeddings, bilingual_dict, k_list, device):
    """
    Evaluates cross-lingual alignment using multiple metrics.
    For each bilingual pair, finds the rank of the correct translation 
    among English embeddings based on cosine similarity.
    Returns:
      - A dictionary mapping each k to precision@k,
      - Mean Reciprocal Rank (MRR),
      - Total number of evaluated pairs.
    """
    # Build a normalized English embedding matrix
    english_matrix, english_words = build_embedding_matrix(english_embeddings, list(english_embeddings.keys()))
    english_matrix = F.normalize(english_matrix, p=2, dim=1)
    
    total = 0
    precision_counts = {k: 0 for k in k_list}
    sum_reciprocal = 0.0

    # For each dictionary pair, compute similarity and rank
    for hindi_word, english_word in tqdm(bilingual_dict, desc="Evaluating alignment", total=len(bilingual_dict)):
        if hindi_word in aligned_hindi_embeddings and english_word in english_embeddings:
            total += 1
            hindi_vec = aligned_hindi_embeddings[hindi_word]
            hindi_vec = F.normalize(hindi_vec, p=2, dim=0)
            # Compute cosine similarities
            similarities = torch.mv(english_matrix, hindi_vec)
            # Get sorted indices (largest similarity first)
            sorted_indices = torch.argsort(similarities, descending=True)
            # Find rank (1-indexed) of the correct English word
            try:
                rank = (sorted_indices == english_words.index(english_word)).nonzero(as_tuple=False).item() + 1
            except Exception:
                rank = None

            # Update precision counters for each k
            if rank is not None:
                for k in k_list:
                    if rank <= k:
                        precision_counts[k] += 1
                sum_reciprocal += 1.0 / rank
            else:
                # If not found, contribute 0 to reciprocal rank.
                sum_reciprocal += 0.0

    precision_at_k = {k: (precision_counts[k] / total if total > 0 else 0) for k in k_list}
    mrr = sum_reciprocal / total if total > 0 else 0

    return precision_at_k, mrr, total

def plot_precision(precision_dict, output_file='visualizations/precision_bar_chart.png'):
    """
    Generates a bar chart comparing precision@k values.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ks = list(precision_dict.keys())
    precisions = [precision_dict[k] * 100 for k in ks]  # as percentage
    plt.figure(figsize=(8, 6))
    plt.bar([str(k) for k in ks], precisions, color='skyblue')
    plt.xlabel('k (Top-k)')
    plt.ylabel('Precision@k (%)')
    plt.title('Precision@k for Cross-lingual Alignment')
    plt.ylim(0, 100)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved precision bar chart to {output_file}")

def generate_report(precision_dict, mrr, total, report_file='evaluation_report.txt'):
    """
    Generates a text report summarizing evaluation results.
    """
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Report for Cross-Lingual Alignment\n")
        f.write("=============================================\n")
        f.write(f"Total evaluated pairs: {total}\n")
        for k, prec in precision_dict.items():
            f.write(f"Precision@{k}: {prec*100:.2f}%\n")
        f.write(f"Mean Reciprocal Rank (MRR): {mrr:.4f}\n")
    print(f"Saved evaluation report to {report_file}")

def visualize_evaluation(aligned_hindi_embeddings, english_embeddings, bilingual_dict, sample_size=200, output_file='visualizations/tsne_evaluation.png'):
    """
    Creates a t-SNE visualization for a sample of bilingual pairs.
    """
    sample_pairs = random.sample(bilingual_dict, min(sample_size, len(bilingual_dict)))
    hindi_vecs, english_vecs, labels = [], [], []
    for hindi_word, english_word in sample_pairs:
        if hindi_word in aligned_hindi_embeddings and english_word in english_embeddings:
            hindi_vecs.append(aligned_hindi_embeddings[hindi_word].detach().cpu().numpy())
            english_vecs.append(english_embeddings[english_word].detach().cpu().numpy())
            labels.append(f"{hindi_word}/{english_word}")
    if len(hindi_vecs) == 0:
        print("Not enough samples for evaluation visualization.")
        return
    data = np.concatenate([np.array(hindi_vecs), np.array(english_vecs)], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:len(hindi_vecs), 0], tsne_result[:len(hindi_vecs), 1], c='red', label='Aligned Hindi')
    plt.scatter(tsne_result[len(hindi_vecs):, 0], tsne_result[len(hindi_vecs):, 1], c='blue', label='English')
    for i, label in enumerate(labels):
        if i % (max(1, len(labels)//10)) == 0:
            plt.annotate(label, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
    plt.legend()
    plt.title("t-SNE Visualization of Evaluation Sample")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved t-SNE evaluation visualization to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate cross-lingual alignment with enhanced metrics, visualization, and report generation."
    )
    parser.add_argument('--aligned_hindi', type=str, required=True, help="Path to aligned Hindi embeddings file (text or .pt)")
    parser.add_argument('--english', type=str, required=True, help="Path to English embeddings file (text or .pt)")
    parser.add_argument('--dict', type=str, required=True, help="Path to bilingual dictionary file (hindi english per line)")
    parser.add_argument('--k_list', type=str, default="1,5,10", help="Comma-separated list of k values for precision@k (e.g., '1,5,10')")
    parser.add_argument('--max_vocab', type=int, default=None, help="Maximum number of words to load from embeddings")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--sample_size', type=int, default=200, help="Sample size for t-SNE visualization")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading aligned Hindi embeddings...")
    aligned_hindi_embeddings = load_embeddings(args.aligned_hindi, max_vocab=args.max_vocab, device=device)
    print(f"Loaded {len(aligned_hindi_embeddings)} aligned Hindi embeddings.")

    print("Loading English embeddings...")
    english_embeddings = load_embeddings(args.english, max_vocab=args.max_vocab, device=device)
    print(f"Loaded {len(english_embeddings)} English embeddings.")

    print("Loading bilingual dictionary...")
    bilingual_dict = load_bilingual_dictionary(args.dict)
    print(f"Loaded {len(bilingual_dict)} dictionary entries.")

    # Parse the comma-separated k values
    k_list = [int(k.strip()) for k in args.k_list.split(',')]

    print(f"Evaluating cross-lingual alignment for k values: {k_list} ...")
    precision_dict, mrr, total = evaluate_alignment(aligned_hindi_embeddings, english_embeddings, bilingual_dict, k_list, device)
    print("Evaluation Results:")
    for k, prec in precision_dict.items():
        print(f"Precision@{k}: {prec*100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Total evaluated pairs: {total}")

    print("Generating precision bar chart...")
    plot_precision(precision_dict, output_file='visualizations/precision_bar_chart.png')

    print("Generating t-SNE visualization for evaluation sample...")
    visualize_evaluation(aligned_hindi_embeddings, english_embeddings, bilingual_dict, sample_size=args.sample_size, output_file='visualizations/tsne_evaluation.png')

    print("Generating evaluation report...")
    generate_report(precision_dict, mrr, total, report_file='evaluation_report.txt')
