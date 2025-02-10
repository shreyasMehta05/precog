#!/usr/bin/env python3
import numpy as np

def filter_with_required(input_path, output_path, required_words, max_words=160000):
    """
    Filter FastText vectors to keep required words plus the most frequent words up to max_words,
    and save them in the standard word2vec text format (with a header line).
    
    Args:
        input_path (str): Path to the original FastText embeddings file.
        output_path (str): Path to save the filtered embeddings.
        required_words (set): Set of words that must be included.
        max_words (int): Maximum number of words to keep.
    """
    vectors = {}
    dim = None

    print("Reading vectors...")
    with open(input_path, 'r', encoding='utf-8') as f:
        # Read the header and extract the dimensionality.
        header = f.readline().strip()
        try:
            # Expect header to be in the format: "vocab_size dimension"
            _, dim_str = header.split()[:2]
            dim = int(dim_str)
        except Exception as e:
            print("Error reading header. Ensure the header is in the format: '<vocab_size> <dimension>'")
            raise e
        
        # First pass: add all required words.
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            vector = parts[1:]
            # Only add if the word is required.
            if word in required_words:
                try:
                    # Convert each component to float and then back to string to ensure valid formatting.
                    vector_floats = [str(float(x)) for x in vector]
                    vectors[word] = vector_floats
                except ValueError:
                    continue
            if len(vectors) >= max_words:
                break
        
        # If we haven't reached max_words, add the most frequent remaining words.
        if len(vectors) < max_words:
            # Rewind file to re-read (skip header)
            f.seek(0)
            next(f)
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                word = parts[0]
                if word not in vectors:
                    vector = parts[1:]
                    try:
                        vector_floats = [str(float(x)) for x in vector]
                        vectors[word] = vector_floats
                    except ValueError:
                        continue
                if len(vectors) >= max_words:
                    break
    
    print(f"Writing {len(vectors)} vectors to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header: number of vectors and dimension.
        f.write(f"{len(vectors)} {dim}\n")
        # Write each vector line in the format: word value1 value2 ... valueN
        for word, vector in vectors.items():
            f.write(f"{word} {' '.join(vector)}\n")
    print("Filtering complete.")

def get_required_words(dict_path, similarity_path):
    """
    Get a set of required target words from a bilingual dictionary and a similarity dataset.
    
    Args:
        dict_path (str): Path to the bilingual dictionary file (each line: source<tab>target).
        similarity_path (str): Path to the similarity dataset (each line: source<tab>target<tab>score).
    
    Returns:
        set: Set of required target words.
    """
    required_words = set()
    
    # Add target words from the bilingual dictionary.
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                _, en_word = parts[:2]
                required_words.add(en_word)
    
    # Add target words from the similarity dataset.
    with open(similarity_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                _, en_word, _ = parts[:3]
                required_words.add(en_word)
    
    return required_words

if __name__ == "__main__":
    # Define file paths (adjust these as needed)
    dict_path = 'data/hi_en_dictionary.txt'
    similarity_path = 'data/similarity_dataset.txt'
    input_path = 'data/english_embeddings.vec'
    output_path = 'data/english_filtered.vec'
    
    print("Gathering required words...")
    required_words = get_required_words(dict_path, similarity_path)
    print(f"Total required words: {len(required_words)}")
    
    filter_with_required(input_path, output_path, required_words, max_words=160000)
