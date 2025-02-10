"""
Optimized Pipeline for GPU-Accelerated Cleaning, Tokenization, and Embedding Generation (English Only)
- Loads and cleans data on GPU using cuDF.
- Uses spaCy (with GPU enabled) for batch tokenization.
- Builds vocabulary and a co-occurrence matrix with a sliding window.
- Applies dimensionality reduction using TruncatedSVD.
- Displays progress/status bars and timing for each step.
"""

import time
import re
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from tqdm import tqdm

# ----------------------------
# Step 1: GPU-Accelerated Data Loading & Cleaning with RAPIDS
# ----------------------------
import cudf

def clean_text_gpu(text_series):
    """
    Use cuDF vectorized string methods to lowercase and remove punctuation.
    """
    clean_series = text_series.str.lower()
    clean_series = clean_series.str.replace(r'[^\w\s]', '', regex=True)
    return clean_series

# Timing the loading and cleaning step
start_time = time.perf_counter()
print("Loading English corpus on GPU...")
# Update the path to your 10K or 300K sentence corpus as needed.
# eng_df = cudf.read_csv("./dataset/raw/eng_10k/eng_news_2024_10k-sentences.txt",
#                        header=None, names=["sentence"])
eng_df = cudf.read_csv("./dataset/raw/eng_300k/eng_news_2024_300k-sentences.txt",
                       header=None, names=["sentence"])
eng_df["clean"] = clean_text_gpu(eng_df["sentence"])
print(f"Corpus loaded and cleaned in {time.perf_counter() - start_time:.2f} seconds.")

# ----------------------------
# Step 2: Tokenization using spaCy with GPU acceleration
# ----------------------------
import spacy

# Enable GPU (if available)
if spacy.prefer_gpu():
    spacy.require_gpu()
nlp_en = spacy.load("en_core_web_sm")  # You can switch to a larger model if desired

def spacy_tokenize_pipe(sentences, batch_size=1000):
    """
    Tokenize sentences using spaCy's pipe for batch processing.
    Returns a list of token lists.
    """
    # n_process=-1 uses all available cores; adjust batch_size as needed.
    docs = list(nlp_en.pipe(sentences, batch_size=batch_size, n_process=1))
    return [[token.text for token in doc] for doc in docs]

start_time = time.perf_counter()
# Use spaCy's pipe for fast, batch tokenization
print("Tokenizing English corpus...")
sentences = eng_df["clean"].to_pandas().tolist()
tokenized_sentences = spacy_tokenize_pipe(sentences, batch_size=1000)
print(f"Tokenization complete in {time.perf_counter() - start_time:.2f} seconds.")


# ----------------------------
# Step 3: Build Vocabulary and Co-Occurrence Matrix
# ----------------------------
def build_vocab_and_cooccurrence(tokenized_sentences, window_size=5):
    """
    Build a vocabulary and a sparse co-occurrence matrix from tokenized sentences.
    Displays progress with tqdm.
    Returns: vocabulary dict and co-occurrence matrix.
    """
    start = time.perf_counter()
    
    # Build vocabulary with progress indication
    word_counter = Counter()
    for tokens in tqdm(tokenized_sentences, desc="Building vocabulary"):
        word_counter.update(tokens)
    vocab = {word: idx for idx, word in enumerate(word_counter.keys())}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize a sparse co-occurrence matrix in LIL format (efficient for incremental updates)
    cooc_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    
    # Build the co-occurrence matrix with a sliding window, with progress bar
    for tokens in tqdm(tokenized_sentences, desc="Building co-occurrence matrix"):
        indices = [vocab[token] for token in tokens if token in vocab]
        for i, center in enumerate(indices):
            start_idx = max(0, i - window_size)
            end_idx = min(len(indices), i + window_size + 1)
            for j in range(start_idx, end_idx):
                if i != j:
                    cooc_matrix[center, indices[j]] += 1.0
    print(f"Vocabulary and co-occurrence matrix built in {time.perf_counter() - start:.2f} seconds.")
    return vocab, cooc_matrix

start_time = time.perf_counter()
print("Processing English corpus...")
vocab_en, cooc_en = build_vocab_and_cooccurrence(tokenized_sentences, window_size=5)
print(f"Corpus processed in {time.perf_counter() - start_time:.2f} seconds.")

# ----------------------------
# Step 4: Dimensionality Reduction via TruncatedSVD
# ----------------------------
d = 300  # Target embedding dimensions
start_time = time.perf_counter()
print("Performing SVD on English co-occurrence matrix...")
svd_en = TruncatedSVD(n_components=d, n_iter=10, random_state=42)
embeddings_en = svd_en.fit_transform(cooc_en)
print(f"SVD complete in {time.perf_counter() - start_time:.2f} seconds.")

# ----------------------------
# Step 5: Evaluation Example - Cosine Similarity
# ----------------------------
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

example_words_en = ["president", "government"]
if all(word in vocab_en for word in example_words_en):
    idx1 = vocab_en[example_words_en[0]]
    idx2 = vocab_en[example_words_en[1]]
    sim_en = cosine_similarity(embeddings_en[idx1], embeddings_en[idx2])
    print(f"Cosine similarity between '{example_words_en[0]}' and '{example_words_en[1]}' (English): {sim_en:.4f}")
else:
    print("One or both example words not found in English vocabulary.")

# ----------------------------
# Optional: Save Embeddings and Vocabulary
# ----------------------------
np.save("eng_embeddings.npy", embeddings_en)
with open("vocab_en.txt", "w", encoding="utf-8") as f_en:
    for word, idx in vocab_en.items():
        f_en.write(f"{word}\t{idx}\n")

print("Processing complete.")
