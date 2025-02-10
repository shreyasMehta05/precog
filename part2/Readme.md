
# Part 2: Cross-Lingual Alignment, Evaluation, and Visualization Report

This document provides a comprehensive overview of our experiments and methods for aligning Hindi and English word embeddings into a unified vector space. The goal is to facilitate cross-lingual natural language processing tasks by ensuring that semantically equivalent words from different languages are mapped close together. In this report, we detail our methodology, alignment techniques, evaluation strategies, visualization methods, and the challenges encountered—particularly those related to memory constraints.

---

## Table of Contents

1. [Overview](#overview)
2. [Methodology and Approach](#methodology-and-approach)
   - [Data Preparation and Similarity Dataset Generation](#data-preparation-and-similarity-dataset-generation)
   - [Alignment Techniques](#alignment-techniques)
      - [Iterative Self-Learning Alignment](#iterative-self-learning-alignment)
      - [Optimal Transport Alignment](#optimal-transport-alignment)
      - [Advanced Alignment with Procrustes Analysis & CCA](#advanced-alignment-with-procrustes-analysis--cca)
   - [Evaluation Strategies](#evaluation-strategies)
   - [Visualization Techniques](#visualization-techniques)
   - [Preprocessing and Memory Optimization](#preprocessing-and-memory-optimization)
3. [Scripts Overview](#scripts-overview)
4. [Usage Instructions](#usage-instructions)
5. [Challenges and Observations](#challenges-and-observations)
6. [Summary](#summary)

---

## Overview

In this part of the project, our primary objective is to learn a transformation that maps Hindi word embeddings (source) to English word embeddings (target), creating a shared embedding space. Such an alignment allows us to:
- Translate words between languages,
- Compare semantic similarities,
- Transfer knowledge from one language to another,
- Evaluate and analyze the inherent biases and semantic relationships across languages.

To achieve this, we use a bilingual dictionary as a guide and employ multiple alignment strategies, including iterative self-learning, optimal transport with entropic regularization, and advanced methods such as Procrustes Analysis and Canonical Correlation Analysis (CCA). Furthermore, we evaluate the quality of the alignment using quantitative metrics and visualize the aligned embeddings to provide a qualitative assessment.

---

## Methodology and Approach

### Data Preparation and Similarity Dataset Generation

**Script: `generate_similarity_dataset.py`**

- **Objective:**  
  Construct a comprehensive similarity dataset containing Hindi–English word pairs. The dataset spans various semantic categories, including basic objects, animals, food and drinks, colors, family relationships, common verbs, time and numbers, body parts, emotions, weather, professions, places, technology, directions, adjectives, and additional categories such as transportation, education, clothing, household items, sports, government/politics, banking/finance, medical terms, nature, kitchen items, musical instruments, shopping, travel/tourism, and legal terms.

- **Details:**  
  - The script defines an extensive list of word pairs along with human-annotated similarity scores.
  - After defining the primary list, additional pairs are appended to extend the dataset.
  - The output is written to a text file (`similarity_dataset.txt`) using UTF‑8 encoding, ensuring that the file contains a header row (with columns `Hindi`, `English`, and `Similarity_Score`) followed by the word pair records.
  
- **Importance:**  
  This dataset is crucial for later stages where we evaluate whether the cross-lingual alignment preserves semantic similarities. By comparing the cosine similarities of aligned vectors with the human-annotated scores, we can quantitatively assess the quality of the alignment.

---

### Alignment Techniques

To map the Hindi embeddings into the English embedding space, we explored several alignment methods:

#### Iterative Self-Learning Alignment

**Script: `alignment_iterative.py`**

- **Concept:**  
  The iterative self-learning alignment method refines an initial transformation matrix by repeatedly updating it based on the nearest neighbor relationships between the source and target embeddings.

- **Process Details:**
  1. **Initial Setup:**  
     - Load Hindi (source) and English (target) embeddings using Gensim’s `KeyedVectors`.
     - Load a bilingual dictionary that maps Hindi words to their English translations.
  
  2. **Matrix Construction:**  
     - Extract matrices \(X\) and \(Y\) from the bilingual word pairs.
     - Each row in these matrices corresponds to the embedding vector of a word from the source and target languages, respectively.
  
  3. **Initial Transformation:**  
     - Compute an initial transformation matrix \(W\) using Singular Value Decomposition (SVD) on the cross-covariance matrix \(M = Y^T X\).
  
  4. **Iterative Refinement:**  
     - In each iteration, transform the source embeddings using the current \(W\).
     - Use cosine similarity (computed in batches to save memory) to determine the \(k\)-nearest neighbors for each source word.
     - Rebuild new source and target matrices based on these nearest neighbors.
     - Recompute \(W\) via SVD using the new matrices.
  
  5. **Output:**  
     - Save the final aligned source embeddings and the transformation matrix.
  
- **Memory Optimization:**  
  - The process employs batch processing and GPU acceleration where possible.
  - Despite these techniques, the iterative method still demands significant memory, leading to prolonged training times (reported up to 28 hours in some cases) and incomplete quantitative evaluation.

#### Optimal Transport Alignment

**Script: `alignment_optimal_transport.py`**

- **Concept:**  
  The optimal transport (OT) method computes a transport plan between the source and target embedding spaces using entropic regularization, which leads to a soft alignment that is both robust and memory-efficient.

- **Process Details:**
  1. **Sampling:**  
     - A subset of bilingual pairs is sampled to reduce the computational burden.
  
  2. **Matrix Construction:**  
     - Construct matrices \(X\) (source) and \(Y\) (target) using the sampled word pairs.
  
  3. **Cost Matrix Calculation:**  
     - Compute a cost matrix \(C\) based on negative cosine similarities, processing the matrix in batches to manage memory usage.
  
  4. **Sinkhorn Algorithm:**  
     - Use the Sinkhorn algorithm (via the `ot` package) with a regularization parameter to compute the optimal transport plan \(P\).
     - Normalize \(P\) so that each row sums to one, and compute the barycenter \(Y_{bar}\) of the target embeddings.
  
  5. **Transformation via Procrustes:**  
     - Apply orthogonal Procrustes Analysis to derive the transformation matrix \(W\) that best aligns \(X\) to \(Y_{bar}\).
     - Transform all source embeddings using \(W\).
  
  6. **Output:**  
     - Save the aligned embeddings and the transformation matrix.
  
- **Memory Optimization:**  
  - This approach is designed to be more memory-efficient by leveraging sampling and processing the cost matrix in manageable batches.
  - It effectively reduces memory overhead compared to the iterative method, while still achieving robust alignment.

#### Advanced Alignment with Procrustes Analysis & CCA

**Scripts: Advanced functions in `CrossLingualAligner` and `AdvancedCrossLingualAligner` classes**

- **Procrustes Analysis:**  
  - Uses SVD to compute an optimal linear transformation that minimizes the Euclidean distance between the transformed source and target embeddings.
  - This method is GPU-accelerated, which significantly speeds up the computation for large matrices.
  
- **Canonical Correlation Analysis (CCA):**  
  - Uses scikit-learn’s CCA implementation to find a shared subspace where the correlation between the source and target embeddings is maximized.
  - Although this method is CPU-based, it is generally more memory-efficient and provides competitive alignment performance.

- **Output:**  
  - Both methods save the aligned embeddings and corresponding transformation matrices, allowing for direct comparison in subsequent evaluations.

---

### Evaluation Strategies

**Script: Evaluation functions in the `CrossLingualEvaluator` class**

- **Word Translation Evaluation:**  
  - Measures precision@k (P@1, P@5, P@10) by checking if the correct English translation appears within the top‑k nearest neighbors of the aligned Hindi embedding.
  
- **Semantic Similarity Evaluation:**  
  - Computes cosine similarities between aligned Hindi vectors and their corresponding English vectors.
  - Uses Spearman correlation and mean absolute error (MAE) to compare these computed similarities against human-annotated similarity scores from the dataset.
  
- **Hubness Evaluation:**  
  - Evaluates the phenomenon of “hubness,” where certain target vectors appear disproportionately in the nearest neighbor lists.
  - Reports metrics such as maximum hub size, mean hub size, standard deviation, and skewness.

Results from these evaluations are saved in a JSON file (`evaluation_results.json`) for further analysis.

---

### Visualization Techniques

**Script: Visualization functions in the `AlignmentVisualizer` class**

- **t‑SNE Visualization:**  
  - Reduces the dimensionality of the aligned embeddings using t‑SNE, allowing for 2D or 3D visualizations.
  - Generates scatter plots where points are color-coded by semantic category and labeled by language (Hindi vs. English).
  
- **Heatmap Visualization:**  
  - Computes cosine similarity matrices for word pairs grouped by semantic concepts.
  - Produces heatmaps that visually represent the similarity between aligned Hindi and English embeddings for each category.
  
- **Output:**  
  - All visualizations are saved as high-resolution PNG images in the specified output directory, which can be used for qualitative analysis and included in presentations or reports.

---

### Preprocessing and Memory Optimization

**Scripts: `fix_vector_file.py` and `filter_with_required.py`**

- **Fixing Vector Files:**  
  - The `fix_vector_file.py` script ensures that embedding files (particularly those from FastText) are correctly formatted.
  - Multi-word tokens are joined with underscores, and the file is reformatted to follow the standard word2vec text format.

- **Filtering Embeddings:**  
  - The `filter_with_required.py` script reduces the size of the embedding files by retaining only “required” words.
  - These required words are determined from a bilingual dictionary and a similarity dataset.
  - Filtering out infrequent or unnecessary words reduces memory usage and speeds up subsequent alignment processes.

These preprocessing steps are vital for managing large-scale embeddings, which otherwise demand significant computational resources.

---

## Scripts Overview

All scripts for Part 2 are contained in the `part2/Tried_Approaches_Incomplete/crossLingualAlignment/scripts/` directory. The key scripts include:

- **Data Generation and Preprocessing:**
  - `generate_similarity_dataset.py`: Generates the Hindi–English similarity dataset.
  - `fix_vector_file.py`: Fixes formatting issues in the embedding files.
  - `filter_with_required.py`: Filters the embeddings to retain only the required vocabulary.
  
- **Alignment Methods:**
  - `alignment_iterative.py`: Implements the iterative self-learning alignment approach.
  - `alignment_optimal_transport.py`: Implements the optimal transport alignment approach.
  - Advanced alignment functions (e.g., Procrustes Analysis and CCA) are included within the `CrossLingualAligner` and `AdvancedCrossLingualAligner` classes.
  
- **Evaluation:**
  - Evaluation functions are implemented in the `CrossLingualEvaluator` class. These include word translation evaluation, semantic similarity assessment, and hubness metrics.
  
- **Visualization:**
  - The `AlignmentVisualizer` class provides functions to generate t‑SNE visualizations and similarity heatmaps, which help to qualitatively assess the alignment.

- **Additional Utility Scripts:**
  - `download.py`: Downloads necessary bilingual dictionaries.
  - Other scripts supporting concept group generation, quantitative evaluation, and visualization are also provided.

---


### Running the Scripts

#### 1. Generate the Similarity Dataset
   ```bash
   python generate_similarity_dataset.py
   ```
   - This script creates the `data/similarity_dataset.txt` file, which contains Hindi–English word pairs with similarity scores.

#### 2. Preprocess Embeddings (Optional)
   ```bash
   python fix_vector_file.py
   python filter_with_required.py
   ```
   - Run these scripts to correct formatting issues and reduce the vocabulary size of the embedding files, thereby lowering memory usage.

#### 3. Alignment Methods

- **Iterative Self-Learning Alignment:**
   ```bash
   python alignment_iterative.py --src_emb path/to/hindi_embeddings.vec --tgt_emb path/to/english_embeddings.vec --dict_path path/to/hi_en_dictionary.txt --output_dir iterative_alignment_output --max_iter 5 --k 5 --batch_size 1000 --use_dict_target
   ```
   - Adjust parameters (iterations, batch size, etc.) based on your system’s memory capacity.

- **Optimal Transport Alignment:**
   ```bash
   python alignment_optimal_transport.py --src_emb path/to/hindi_embeddings.vec --tgt_emb path/to/english_embeddings.vec --dict_path path/to/hi_en_dictionary.txt --output_dir ot_alignment_output --sample_size 5000 --batch_size 1000 --reg 0.05
   ```

- **Advanced Alignment (Procrustes & CCA):**
   ```bash
   python advanced_alignment.py
   ```
   - This script (or section within the AdvancedCrossLingualAligner class) will perform both Procrustes Analysis and CCA alignments and save the aligned embeddings and transformation matrices.

#### 4. Evaluate Alignment Quality

- **Run Evaluation:**
   ```bash
   python evaluation_script.py
   ```
   - This command executes the `CrossLingualEvaluator` class, which computes word translation accuracy (precision@k), semantic similarity (Spearman correlation and MAE), and hubness metrics. Results are saved to `evaluation_results.json`.

#### 5. Visualize Aligned Embeddings

- **Generate Visualizations:**
   ```bash
   python visualization_script.py
   ```
   - This command invokes the `AlignmentVisualizer` class, generating t‑SNE plots and heatmaps for each alignment method. Visualizations are saved in the `visualization_results` directory.

---

## Challenges and Observations

- **Memory Constraints:**  
  The iterative self-learning alignment method, even with batch processing and GPU acceleration, demanded significant memory and long training times (up to 28 hours), which restricted complete quantitative evaluation.
  
- **Optimal Transport Efficiency:**  
  The optimal transport method, utilizing sampling and batch processing, proved to be more memory-efficient and faster while still producing robust alignment results.

- **Advanced Methods:**  
  Procrustes Analysis (with GPU acceleration) and CCA (CPU-based) delivered competitive alignment performance. However, the size and high dimensionality of modern embeddings continue to pose challenges.
  
- **Preprocessing Impact:**  
  Preprocessing scripts that fix vector formatting and filter embeddings to include only required words helped reduce memory overhead significantly. Nevertheless, processing very large embedding sets remains a resource-intensive task.

- **Evaluation Insights:**  
  Quantitative evaluations (translation accuracy, semantic similarity, hubness) provide important insights into the alignment quality. Visualizations further help in understanding the preservation of semantic relationships across languages.

---

## Summary

This report summarizes our efforts in aligning Hindi and English word embeddings using several state-of-the-art techniques:

- **Data Preparation:**  
  Generated a rich Hindi–English similarity dataset for evaluation purposes.

- **Alignment Techniques:**  
  Implemented multiple alignment methods:
  - **Iterative Self-Learning Alignment:** A promising approach with iterative refinement, yet constrained by memory and time.
  - **Optimal Transport Alignment:** A more memory-efficient method that leverages sampling and batch processing.
  - **Advanced Methods (Procrustes & CCA):** Robust approaches that offer competitive performance.

- **Evaluation and Visualization:**  
  Employed rigorous evaluation metrics (word translation, semantic similarity, hubness) and visualized results via t‑SNE and heatmaps to assess the quality of the alignment.

- **Memory Optimization:**  
  Utilized preprocessing steps (vector file fixing and filtering) and batch processing to mitigate memory issues, though resource constraints remain a significant challenge.

This comprehensive documentation and the associated scripts provide a detailed guide to reproducing our cross-lingual alignment experiments, evaluating their performance, and visualizing the results. We hope that this report serves as a valuable resource for further research and development in cross-lingual natural language processing.

