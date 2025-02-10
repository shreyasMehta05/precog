# **Dense Word Representations via Co-Occurrence Matrices**

## 🚀 **Introduction**

In the world of **Natural Language Processing (NLP)**, word embeddings are pivotal for capturing the semantic meanings of words by transforming them into continuous vector spaces. This project focuses on the construction of dense word representations using **co-occurrence matrices** and **dimensionality reduction techniques**. The core idea behind this approach is that words appearing in similar contexts tend to share related meanings. This report dives into the methodology, experimental setup, and performance evaluation of the word embeddings generated through this approach.

## 🧪 **Testing for System Requirements**

The pipeline follows a structured approach, ensuring optimal results and efficiency:

### 1. **Data Preprocessing**
   - We begin with a **large English corpus** containing **300K sentences**.
   - Text is preprocessed by converting it to **lowercase** and removing **punctuation** through basic tokenization.
   - The dataset is loaded and processed with **GPU acceleration** wherever applicable.

### 2. **Tokenization**
   - The text is tokenized using the **spaCy NLP library**.
   - To improve efficiency, **batch tokenization** is applied, though **sequential tokenization** is used due to GPU compatibility limitations.

### 3. **Vocabulary and Co-Occurrence Matrix Construction**
   - A **vocabulary** is built by identifying unique tokens and their frequencies.
   - A **sparse co-occurrence matrix** is created using a sliding window approach with a window size of **5**.
   - This matrix captures word relationships based on contextual co-occurrence in the corpus.

### 4. **Dimensionality Reduction**
   - **Truncated Singular Value Decomposition (SVD)** is used to reduce the high-dimensional co-occurrence matrix.
   - The resulting word vectors are projected into a **lower-dimensional space** (d = 300), allowing us to retain the most important semantic relationships while reducing noise.

### 5. **Evaluation**
   - The cosine similarity between word pairs is computed to assess the **semantic proximity**.
   - Example analysis: **Cosine similarity** between “president” and “government.”

---

## 🧑‍💻 **Experimental Setup**

### **Hardware Configuration**
- **Operating System:** Ubuntu on WSL (Windows Subsystem for Linux)
- **GPU:** NVIDIA GeForce RTX 3050 with **CUDA 12.x** support
- **Acceleration:**
  - **Data Processing:** Accelerated with **cuDF** for optimized computation.
  - **Tokenization:** Utilizes **spaCy's GPU capabilities**, though tokenization is sequential due to compatibility constraints.

---

## 📊 **Performance Analysis**

### **Small-Scale Corpus (10K Sentences)**

| **Step**                           | **Time Taken (s)** |
|-------------------------------------|--------------------|
| Data Loading & Cleaning             | 1.33               |
| Tokenization                        | 14.61              |
| Vocabulary & Co-Occurrence Matrix   | 2.97               |
| SVD                                  | 3.78               |
| **Total**                           | **22.69**          |

- **Key Insight**: The small-scale corpus enables quick experimentation and validation of the approach.
- **Cosine Similarity (president, government)**: **0.7652**

### **Full Corpus (300K Sentences)**

| **Step**                           | **Time Taken (s)** |
|-------------------------------------|--------------------|
| Data Loading & Cleaning             | 1.54               |
| Tokenization                        | 332.33             |
| Vocabulary & Co-Occurrence Matrix   | 99.67              |
| SVD                                  | 100.73             |
| **Total**                           | **534.27**         |

- **Key Insight**: The tokenization process dominates the runtime, particularly when processing large datasets.
- **Vocabulary Size**: **359,474 unique words**
- **Cosine Similarity (president, government)**: **0.8690**, demonstrating improved semantic relationships as the dataset size increases.

---

## 🚀 **Next Steps: Actual Implementation**

This initial phase aimed to test the feasibility of the approach and verify that the results are promising. Moving forward, the plan is to:

- **Optimize the pipeline** for improved **efficiency** and **scalability**, particularly addressing the bottleneck in the tokenization step.
- Explore advanced methods for **parallelization** and **distributed computing** to handle even larger corpora efficiently.

---

## 🌟 **Conclusion**

This project demonstrates the potential of using **co-occurrence matrices** combined with **dimensionality reduction** to generate dense word embeddings. While the initial results are promising, there is room for optimization and further experimentation to fully leverage the power of large-scale datasets and computational resources. The next steps will focus on refining the process to ensure high performance across even larger datasets and more complex NLP tasks.

