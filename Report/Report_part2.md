
# Part 2: Cross-Lingual Alignment, Evaluation, and Visualization Report 🚀

This document provides a comprehensive overview of our experiments and methods for aligning Hindi and English word embeddings into a unified vector space. Our ultimate goal is to enable effective cross-lingual NLP by ensuring that semantically equivalent words from different languages are mapped closely together. In this report, we detail our methodology, alignment techniques, evaluation strategies, visualization methods, and the challenges encountered—particularly with memory and resource constraints.

---

## Table of Contents 📂

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

## Overview 🌟

In this part of the project, our primary objective is to learn a transformation that maps Hindi word embeddings (source) to English word embeddings (target), thereby creating a shared embedding space. This unified space allows us to:
- Translate words between languages 🌐,
- Compare semantic similarities 🤝,
- Transfer knowledge across languages 🔄,
- Evaluate and analyze biases and semantic relationships.

We use a bilingual dictionary as our guide and experiment with multiple alignment strategies, including iterative self‑learning, optimal transport with entropic regularization, and advanced methods such as Procrustes Analysis and Canonical Correlation Analysis (CCA). In addition, we evaluate the quality of these alignments through quantitative metrics and visualize the results to offer a qualitative assessment.

---

## Methodology and Approach 🛠️

### Data Preparation and Similarity Dataset Generation 📊

**Script: `generate_similarity_dataset.py`**

- **Objective:**  
  To create a comprehensive Hindi–English similarity dataset covering a wide range of semantic categories—from basic objects and animals to emotions, professions, and more.
  
- **Details:**  
  - A rich list of word pairs along with similarity scores is defined.
  - Additional word pairs from various domains (e.g., transportation, education, clothing) are appended to extend the dataset.
  - The final dataset is written to a UTF‑8 encoded text file (`similarity_dataset.txt`) with a header row (`Hindi`, `English`, `Similarity_Score`).

- **Impact:**  
  This dataset is essential for evaluating whether the alignment preserves semantic similarities between languages, as we later compare cosine similarities of aligned vectors against human-annotated scores.

---

### Alignment Techniques 🔄

We experimented with several alignment methods to map Hindi embeddings into the English embedding space:

#### Iterative Self-Learning Alignment 🔁

**Script: `alignment_iterative.py`**

- **Concept:**  
  This method refines an initial transformation matrix by iteratively updating it based on nearest neighbor searches between the source and target embeddings.

- **Process:**
  1. **Initial Setup:**  
     - Load Hindi and English embeddings using Gensim’s `KeyedVectors`.
     - Load a bilingual dictionary mapping Hindi words to their English translations.
  
  2. **Matrix Construction:**  
     - Construct matrices \(X\) (source) and \(Y\) (target) from the bilingual pairs.
  
  3. **Initial Alignment:**  
     - Compute an initial transformation matrix \(W\) using SVD on the cross‑covariance matrix \(M = Y^T X\).
  
  4. **Iterative Refinement:**  
     - Transform source embeddings using the current \(W\).
     - Compute cosine similarities in batches to identify the \(k\)-nearest neighbors.
     - Update the transformation matrix \(W\) using SVD on the updated matrices.
  
  5. **Output:**  
     - Save the final aligned embeddings and the transformation matrix.
  
- **Memory Optimization:**  
  Batch processing and GPU acceleration were implemented. However, even with these strategies, this method required extensive memory and sometimes took up to 28 hours, making complete quantitative evaluation challenging.

#### Optimal Transport Alignment 🚚

**Script: `alignment_optimal_transport.py`**

- **Concept:**  
  Uses Optimal Transport (OT) with entropic regularization to compute a soft alignment between the embedding spaces.

- **Process:**
  1. **Sampling:**  
     - Sample a subset of bilingual pairs to reduce computational load.
  
  2. **Matrix Construction:**  
     - Build matrices \(X\) and \(Y\) using the sampled pairs.
  
  3. **Cost Matrix Calculation:**  
     - Compute a cost matrix \(C\) (negative cosine similarity) in batches.
  
  4. **Sinkhorn Algorithm:**  
     - Use the Sinkhorn algorithm to compute an optimal transport plan \(P\).
     - Normalize \(P\) and compute the barycenter \(Y_{bar}\) of the target embeddings.
  
  5. **Transformation:**  
     - Compute the transformation matrix \(W\) via orthogonal Procrustes Analysis.
     - Transform the source embeddings with \(W\).
  
  6. **Output:**  
     - Save the aligned embeddings and transformation matrix.
  
- **Memory Optimization:**  
  This approach is more memory‑efficient due to sampling and batch processing, significantly reducing the memory overhead compared to the iterative method.

#### Advanced Alignment with Procrustes Analysis & CCA 🔬

**Scripts: Advanced functions within `CrossLingualAligner` and `AdvancedCrossLingualAligner` classes**

- **Procrustes Analysis:**  
  - Computes an optimal linear transformation using SVD, accelerated by GPU, to minimize the distance between source and target embeddings.
  
- **Canonical Correlation Analysis (CCA):**  
  - Uses scikit‑learn’s CCA to learn a shared subspace, maximizing the correlation between the two languages' embeddings.
  
- **Output:**  
  - Saves the aligned embeddings and corresponding transformation matrices.
  
- **Memory Optimization:**  
  Procrustes analysis leverages GPU acceleration, while CCA, though CPU‑based, is designed to be more memory‑efficient.

---

### Evaluation Strategies 🔍

**Implemented in the `CrossLingualEvaluator` class**

- **Word Translation Evaluation:**  
  - Measures precision@k (P@1, P@5, P@10) by checking if the correct English translation is within the top‑k nearest neighbors of the aligned Hindi vector.
  
- **Semantic Similarity Evaluation:**  
  - Computes cosine similarity between aligned Hindi and English vectors.
  - Uses Spearman correlation and mean absolute error (MAE) to compare the computed similarities with human-annotated scores from our dataset.
  
- **Hubness Evaluation:**  
  - Evaluates the “hubness” phenomenon (certain target vectors appearing too frequently as nearest neighbors).
  - Reports metrics such as maximum hub size, mean hub size, standard deviation, and skewness.
  
Results are saved in a JSON file (`evaluation_results.json`) for later analysis.

---

### Visualization Techniques 🎨

**Implemented in the `AlignmentVisualizer` class**

- **t‑SNE Visualization:**  
  - Reduces the dimensionality of aligned embeddings using t‑SNE.
  - Generates scatter plots with points color-coded by semantic category and language (Hindi vs. English).
  
- **Heatmap Visualization:**  
  - Computes cosine similarity matrices for concept groups.
  - Produces heatmaps that visually depict cross‑lingual similarity for each semantic category.
  
- **Output:**  
  - High-resolution PNG images are saved to the specified output directory for qualitative assessment and reporting.

---

### Preprocessing and Memory Optimization 💾

**Scripts: `fix_vector_file.py` and `filter_with_required.py`**

- **Fixing Vector Files:**  
  - Corrects formatting issues in FastText vector files by joining multi‑word tokens with underscores.
  
- **Filtering Embeddings:**  
  - Filters the large embedding files to retain only “required” words determined from the bilingual dictionary and similarity dataset.
  - This significantly reduces the vocabulary size and memory footprint, although challenges with Word2Vec files persisted despite these efforts.
  
- **Personal Struggles:**  
  - I faced substantial challenges with RAM and overall memory usage during these experiments.
  - Despite filtering and attempting to download pre‑processed vectors, handling large Word2Vec files was particularly problematic.
  - I also created various metrics and datasets (using scripts like `concept_group_generator.py` and `generate_similarity_dataset.py`) to help evaluate the alignments.
  - Even with these preprocessing steps, the sheer scale of the data meant that memory issues were an ongoing obstacle.

---

## Scripts Overview 📑

All scripts for Part 2 are located in the `part2/Tried_Approaches_Incomplete/crossLingualAlignment/scripts/` directory. Key scripts include:

- **Data Generation and Preprocessing:**
  - `generate_similarity_dataset.py`: Generates a comprehensive Hindi–English similarity dataset.
  - `fix_vector_file.py`: Fixes formatting issues in FastText vector files.
  - `filter_with_required.py`: Filters embeddings to reduce vocabulary size and memory usage.
  
- **Alignment Methods:**
  - `alignment_iterative.py`: Implements the iterative self‑learning alignment method.
  - `alignment_optimal_transport.py`: Implements the optimal transport alignment method.
  - Advanced alignment functions (Procrustes Analysis & CCA) are part of the `CrossLingualAligner` and `AdvancedCrossLingualAligner` classes.
  
- **Evaluation:**
  - The `CrossLingualEvaluator` class evaluates word translation, semantic similarity, and hubness.
  
- **Visualization:**
  - The `AlignmentVisualizer` class creates t‑SNE plots and similarity heatmaps.
  
- **Utility Scripts:**
  - `download.py`: Downloads required bilingual dictionaries.
  - Additional scripts support concept group generation, quantitative evaluation, and visualization.

---

## Usage Instructions 📝

### Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts

#### 1. Generate the Similarity Dataset
   ```bash
   python generate_similarity_dataset.py
   ```
   - This will create `data/similarity_dataset.txt` with Hindi–English word pairs and similarity scores.

#### 2. Preprocess Embeddings (Optional)
   ```bash
   python fix_vector_file.py
   python filter_with_required.py
   ```
   - Use these scripts to fix formatting issues and filter the embedding files to reduce memory usage.

#### 3. Run Alignment Methods

- **Iterative Self‑Learning Alignment:**
   ```bash
   python alignment_iterative.py --src_emb path/to/hindi_embeddings.vec --tgt_emb path/to/english_embeddings.vec --dict_path path/to/hi_en_dictionary.txt --output_dir iterative_alignment_output --max_iter 5 --k 5 --batch_size 1000 --use_dict_target
   ```
   - Adjust parameters (iterations, batch size, etc.) based on your available resources.

- **Optimal Transport Alignment:**
   ```bash
   python alignment_optimal_transport.py --src_emb path/to/hindi_embeddings.vec --tgt_emb path/to/english_embeddings.vec --dict_path path/to/hi_en_dictionary.txt --output_dir ot_alignment_output --sample_size 5000 --batch_size 1000 --reg 0.05
   ```

- **Advanced Alignment (Procrustes & CCA):**
   ```bash
   python advanced_alignment.py
   ```
   - This script (or the corresponding functions within the AdvancedCrossLingualAligner class) runs both Procrustes Analysis and CCA and saves the results in the specified output directory.

#### 4. Evaluate Alignment Quality

- **Run Evaluation:**
   ```bash
   python evaluation_script.py
   ```
   - This executes the `CrossLingualEvaluator` class, computing word translation accuracy (P@k), semantic similarity (Spearman correlation, MAE), and hubness metrics. Results are saved to `evaluation_results.json`.

#### 5. Visualize Aligned Embeddings

- **Generate Visualizations:**
   ```bash
   python visualization_script.py
   ```
   - This command invokes the `AlignmentVisualizer` class to generate t‑SNE plots and heatmaps, which are saved in the `visualization_results` directory.

---

## Challenges and Observations ⚠️

- **Memory Constraints:**  
  The iterative self‑learning alignment method, even with batch processing and GPU acceleration, demanded excessive memory and prolonged training times (up to 28 hours in some cases). These constraints limited the full quantitative evaluation of this approach.
  
- **Optimal Transport Efficiency:**  
  Sampling and batch processing in the optimal transport method helped reduce memory overhead, making it significantly more efficient while still delivering robust alignment results.
  
- **Advanced Methods:**  
  Procrustes Analysis (with GPU acceleration) and CCA (CPU-based) performed competitively. However, the high dimensionality and large size of the embeddings remain a persistent challenge.
  
- **Preprocessing Struggles:**  
  I faced significant issues with RAM and memory when working with large Word2Vec files. Despite efforts to filter and preprocess the embeddings (using scripts like `fix_vector_file.py` and `filter_with_required.py`), the challenges persisted. Additionally, downloading pre-processed vectors and checking the filtered outputs did not fully resolve the memory limitations.
  
- **Metric Creation and Dataset Generation:**  
  To support evaluation, I created various metrics and datasets using additional scripts such as `concept_group_generator.py` and `generate_similarity_dataset.py`. These tools were invaluable in quantitatively assessing alignment quality, although they added extra complexity to the workflow.

---

## Summary 📋

This Part 2 report summarizes our comprehensive approach to cross‑lingual alignment between Hindi and English word embeddings:

- **Data Preparation:**  
  A rich Hindi–English similarity dataset was generated to evaluate semantic consistency.

- **Alignment Techniques:**  
  Multiple methods were implemented:
  - **Iterative Self‑Learning Alignment:** A promising but resource-intensive approach.
  - **Optimal Transport Alignment:** A more memory‑efficient method that leverages sampling and batch processing.
  - **Advanced Methods (Procrustes Analysis & CCA):** Robust techniques that deliver competitive alignment results.

- **Evaluation and Visualization:**  
  We used quantitative metrics (word translation, semantic similarity, hubness) and visualizations (t‑SNE plots, heatmaps) to assess the quality of the alignments.

- **Memory Optimization:**  
  Preprocessing steps, such as fixing vector file formatting and filtering embeddings to a reduced vocabulary, helped mitigate memory usage, although resource constraints remained a significant challenge throughout the project.

This documentation and the accompanying scripts provide a detailed guide for reproducing our cross‑lingual alignment experiments, evaluating their performance, and visualizing the outcomes. Despite considerable struggles with memory and RAM limitations, these efforts lay a strong foundation for further research and development in cross‑lingual natural language processing.

Happy aligning and analyzing! 🎉
 
---