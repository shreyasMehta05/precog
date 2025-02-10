# Language Representations: Dense Word Embeddings and Cross-Lingual Alignment

Language is a rich tapestry of meaning and structure, and understanding it computationally remains one of the great challenges in artificial intelligence. In this project, we explore the science of word embeddings—a method to represent words in a continuous, high-dimensional space that captures their meanings. Through this work, we generate dense word representations from large text corpora, evaluate their quality, and work on cross-lingual alignment between English and Hindi embeddings.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Tasks](#project-tasks)
   - [Part 1: Dense Representations](#part-1-dense-representations)
   - [Part 2: Cross-Lingual Alignment](#part-2-cross-lingual-alignment)
   - [Bonus Task: Harmful Associations](#bonus-task-harmful-associations)
   - [Paper Reading Task](#paper-reading-task)
3. [Directory Structure](#directory-structure)
4. [Usage Instructions](#usage-instructions)
5. [Results & Observations](#results--observations)

---

## Overview

This repository documents a comprehensive project that delves into various aspects of language representation. The project is divided into several key parts:

- **Dense Representations:** Generating and evaluating dense word embeddings from large text corpora using co-occurrence matrices and dimensionality reduction techniques.
- **Cross-Lingual Alignment:** Aligning pre-trained monolingual embeddings for English and Hindi to enable cross-lingual knowledge transfer.
- **Harmful Associations:** Quantitatively evaluating harmful word associations in both static and contextual embeddings.
- **Paper Reading Task:** A critical review of literature on language models, supported by a presentation (PPT) and an accompanying video.

---

## Project Tasks

### Part 1: Dense Representations

*“You shall know a word by the company it keeps”* — J.R. Firth

The goal in Part 1 is to generate dense word embeddings by:
- **Constructing a Co-occurrence Matrix:** Converting a large text corpus (minimum 300K sentences) into a numerical co-occurrence matrix. The matrix records how frequently pairs of words occur together within a specified context window.
- **Dimensionality Reduction:** Reducing the size of the NxN co-occurrence matrix to a more manageable N×d matrix using techniques like truncated Singular Value Decomposition (SVD). The optimal window size and dimensionality (d) are determined through experimentation.
- **Quality Evaluation:** Evaluating the quality of the embeddings using cosine similarity, clustering, and visualization techniques (e.g., t-SNE, PCA). Additional evaluation resources like SimLex-999 and WordSimilarity-353 are leveraged to benchmark the performance.

### Part 2: Cross-Lingual Alignment

This part extends the analysis to two languages—English and Hindi. Key objectives include:
- **Downloading Pre-Trained Embeddings:** Obtaining pre-trained monolingual embeddings for both English and Hindi.
- **Alignment Techniques:** Learning transformations (e.g., using Procrustes Analysis, Canonical Correlation Analysis, and Optimal Transport with entropic regularization) to align the embeddings across languages.
- **Quantitative Evaluation:** Assessing the effectiveness of the alignment through quantitative metrics. Advanced experiments were attempted (including memory-efficient iterative self-learning alignment), but due to resource and time constraints (training taking up to 28 hours with RAM limitations), the iterative method did not complete the final quantitative evaluation.

*Advanced Note:* I attempted some advanced methods in cross-lingual alignment using custom datasets (created with the help of Claude and Deepseek, which performed very well in generating data for multiple modalities). The following approaches were successful:
- Alignment using Procrustes Analysis with GPU acceleration.
- Alignment using Canonical Correlation Analysis (CCA) via scikit-learn (CPU-based).
- Alignment using Optimal Transport with entropic regularization.

The iterative self-learning alignment method, however, failed due to resource constraints.

### Bonus Task: Harmful Associations

This task focuses on evaluating potentially harmful associations present in word embeddings:
- **Static Embeddings:** Analysis of static word embeddings (e.g., word2vec, GloVe, FastText) for harmful associations.
- **Contextual Models:** Evaluation of harmful biases in contextual models (such as BERT) using masked language modeling.
- A quantitative evaluation regimen was developed to measure bias in these models.

### Paper Reading Task

A comprehensive paper reading task was also undertaken. This involved:
- Critically reading and summarizing a relevant research paper.
- Creating a presentation (PPT) and a video to encapsulate the paper’s insights and limitations.
- Documenting the process and including reflections on the methodologies and findings.

---

## Directory Structure

Below is an overview of the main directory structure and key files:

```
precog/
├── Main_flow.png                      # Diagram of the overall workflow
├── Readme.md                          # This README file
├── Results_Part1_Concise              # Results and analysis for Part 1 (Dense Representations)
├── Part1.md                           # Detailed documentation for Part 1 tasks
├── Report                             # Contains Reports for better understanding kindly open the link shared in the form
├── english/                          # Scripts and data related to English language Part - 1
├── hindi/                            # Scripts and data related to Hindi language Part-1
├── imgs/                             # Temporary images for html rendering
├── part2/                            # Contains files for cross-lingual alignment (Part 2)
│   └── Tried_Approaches_Incomplete/
│       └── crossLingualAlignment/
│           └── scripts/              # Various scripts for alignment approaches:
│               ├── alignment_iterative.py
│               ├── alignment_optimal_transport.py
│               ├── check_vectors.py
│               ├── concept_group_generator.py
│               ├── download.py
│               ├── filter_en_vectors.py
│               ├── generate_similarity_dataset.py
│               ├── implementation_1.py
│               ├── implementation_2.py
│               ├── quantitative_evaluation.py
│               └── visualisation_evaluation.py
├── paper_reading_task_link.txt        # Link to the paper reading task resources
├── part3/                            # Bonus task: Harmful Associations
├── test/                             # Testing scripts for gpu and time analysis for 1st part
├── requirements.txt                  # List of required Python packages
```

- **Main_flow.png:** Visual representation of the overall project workflow.
- **Part1.md & Report:** Documentation for Part 1, outlining the methodology, experiments, and results for dense representations.
- **english/ & hindi/:** Directories containing scripts and data for processing English and Hindi corpora respectively.
- **part2/:** Contains all scripts related to cross-lingual alignment, including both successful and incomplete approaches.
- **paper_reading_task_link.txt:** Provides links and references to the paper reading task resources.
- **requirements.txt:** Lists all the Python dependencies required for the project.

---

## Usage Instructions

1. **Setup:**
   - Clone the repository.
   - Navigate to the project directory.
   - Install the necessary dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Part 1 (Dense Representations):**
   - Use the scripts in the `english/` and `hindi/` folders to process the respective corpora.
   - Refer to `Part1.md` for detailed instructions on constructing the co-occurrence matrix, performing dimensionality reduction, and evaluating the embeddings.
   - Analyze the results in `Results_Part1_Concise`.

3. **Part 2 (Cross-Lingual Alignment):**
   - Navigate to the `part2/Tried_Approaches_Incomplete/crossLingualAlignment/scripts/` directory.
   - Explore and run the various alignment scripts:
     - **Procrustes Analysis:** `implementation_1.py` (or similar) for GPU-accelerated alignment.
     - **Canonical Correlation Analysis (CCA):** Check the corresponding script using scikit-learn.
     - **Optimal Transport:** `alignment_optimal_transport.py` for memory-efficient alignment.
     - **Iterative Alignment:** `alignment_iterative.py` (not fully successful due to resource constraints).
   - Use `quantitative_evaluation.py` and `visualisation_evaluation.py` for assessing the alignment quality.

4. **Bonus Task (Harmful Associations):**
   - Review and run the scripts for evaluating harmful associations in static and contextual embeddings.
   - Compare results across different models and techniques.

5. **Paper Reading Task:**
   - Access the paper reading task details from `paper_reading_task_link.txt`.
   - Review the accompanying PPT and video for insights into the capabilities and limitations of language models.

---

## Results & Observations

- **Dense Representations (Part 1):**
  - Successfully constructed a co-occurrence matrix from an English corpus (minimum 300K sentences) and reduced its dimensionality using SVD.
  - Evaluations using cosine similarity, clustering, and visualization confirmed the quality of the embeddings.

- **Cross-Lingual Alignment (Part 2):**
  - Pre-trained monolingual embeddings for English and Hindi were aligned using multiple techniques.
  - **Successful Approaches:**
    - Procrustes Analysis with GPU acceleration.
    - Canonical Correlation Analysis (CCA) using scikit-learn.
    - Optimal Transport with entropic regularization.
  - **Challenges:**
    - The memory-efficient iterative self-learning alignment did not complete quantitative evaluation due to time and RAM constraints (training took approximately 28 hours).

- **Bonus Task:**
  - Evaluated harmful associations in both static and contextual embeddings.
  - Explored quantitative methods to assess gender bias and other harmful associations, leveraging external datasets and literature.

- **Paper Reading Task:**
  - Completed an in-depth paper reading, accompanied by a presentation (PPT) and video, discussing the reasoning and limitations of current language models.

## Usage Instructions

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


Overall, all planned tasks were completed, including the bonus task. Additionally, advanced experiments in cross-lingual alignment were attempted, although some methods could not be fully executed due to resource limitations.

---

