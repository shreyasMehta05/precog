# Report on Static-Word-Embeddings Bias Evaluation 📚

This report details the process and results of evaluating gender bias in static word embeddings using the GoogleNews vectors. The evaluation uses a WEAT‐inspired approach to quantify how two groups of target words (representing professions) are differentially associated with two groups of attribute words (representing gender). This analysis helps us understand the stereotypical associations that may be encoded in these embeddings.

---

## 1. Overview

**Command Executed:**  
```bash
python scripts/static_bias.py --model_path data/models/GoogleNews-vectors-negative300.bin.gz
```

**Results Obtained:**
- **Effect Size:** 1.679  
- **Sample Size:** 8  
- **Target Set 1 Mean Differential Association:** 0.008  
- **Target Set 2 Mean Differential Association:** -0.160  

These numbers summarize the bias measured between the two sets of target words relative to two gendered attribute sets.

*Emoji Tip: 📊*

---

## 2. Script Description and Methodology

The Python script follows a series of well-defined steps to evaluate bias in the word embeddings:

### 2.1 Loading the Embeddings 🤖
- **Library:**  
  The script uses the `gensim` library to load the pre-trained GoogleNews word vectors from the file `GoogleNews-vectors-negative300.bin.gz`.
- **Process:**  
  A console message is printed to confirm that the model is being loaded successfully.

### 2.2 Defining Target and Attribute Sets 🎯
- **Target Set 1:**  
  `['doctor', 'engineer', 'scientist', 'programmer']`  
  *These words are considered either neutral or stereotypically male-associated.*
- **Target Set 2:**  
  `['nurse', 'teacher', 'librarian', 'homemaker']`  
  *These words are often stereotypically linked to female roles.*
- **Attribute Set 1 (Male):**  
  `['he', 'man', 'his', 'male']`
- **Attribute Set 2 (Female):**  
  `['she', 'woman', 'her', 'female']`

*Emoji Tip: 🎯👨‍⚕️👩‍🏫*

### 2.3 Computing Differential Associations 🔍
For every target word:
1. **Embedding Retrieval:**  
   The script retrieves the corresponding word embedding (if the word exists in the vocabulary).
2. **Cosine Similarity Calculation:**  
   It computes the cosine similarity between the target word and each word in both attribute sets.
3. **Mean Similarity:**  
   - Mean similarity with **Attribute Set 1** is calculated.
   - Mean similarity with **Attribute Set 2** is calculated.
4. **Differential Association:**  
   The differential association for each target word is computed as:  
   \[
   s(w, A_1, A_2) = \text{mean}_{a \in A_1} \cos(w, a) - \text{mean}_{b \in A_2} \cos(w, b)
   \]
   [//](../imgs/Eqn-last.png)

*Emoji Tip: 📐*

### 2.4 Calculating the WEAT Effect Size 📏
- **Grouping:**  
  Differential associations are grouped by target sets.
- **Mean Calculation:**  
  - **Target Set 1 Mean:** 0.008  
  - **Target Set 2 Mean:** -0.160
- **Effect Size Formula:**  
  The overall effect size is computed as:
  \[
  d = \frac{\text{mean}(s(T_1)) - \text{mean}(s(T_2))}{\text{std}(s(T_1 \cup T_2))}
  \]
  In this evaluation, **d = 1.679**, which is considered large (typically, Cohen’s d > 0.80 is large).
- **Sample Size:**  
  The analysis is based on a total of 8 words (4 per target set).

*Emoji Tip: 📏🔢*

### 2.5 Missing Words Check ✅
- The script also checks for any words that are not found in the vocabulary.
- **Outcome:**  
  All words were present, so no words were missing.

---

## 3. Interpretation of the Results

The key findings from the evaluation are as follows:

### Effect Size and What It Means
- **Effect Size (1.679):**  
  A high effect size suggests that there is a strong differential association between the two target sets.  
  - **Target Set 1 (0.008):**  
    Indicates that words like "doctor" and "engineer" are nearly equally associated with both male and female attribute sets.
  - **Target Set 2 (-0.160):**  
    Indicates that words like "nurse" and "teacher" tend to be more strongly associated with the female attribute set.

*Emoji Tip: 📊💥*

### Implications
- **Stereotypical Associations:**  
  The large effect size implies that the static GoogleNews embeddings encode strong stereotypical associations. For instance, professions such as "nurse" might be more closely linked to female attributes, while "engineer" might lean toward male attributes.
- **Downstream Impact:**  
  Such biases, if left unchecked, can propagate into downstream applications, potentially reinforcing harmful stereotypes in AI systems.

---

## 4. Summary Table

The following table summarizes the key evaluation metrics:

| Metric                                        | Value   | Interpretation                                                                                     |
|-----------------------------------------------|---------|----------------------------------------------------------------------------------------------------|
| **Effect Size**                               | 1.679   | Indicates a strong differential association between target sets.                                 |
| **Sample Size**                               | 8       | Analysis based on 8 target words (4 per set); a larger sample could yield more robust results.      |
| **Target Set 1 Mean Differential Association**| 0.008   | Suggests nearly equal association with both attribute sets.                                      |
| **Target Set 2 Mean Differential Association**| -0.160  | Indicates a skew toward female attribute words for this set.                                      |

*Emoji Tip: 📝✅*

---

## 5. Discussion and Limitations

### Bias Measurement Approach 🔍
- **Inspired by WEAT:**  
  This methodology is based on the Word Embedding Association Test (WEAT) by Caliskan et al. (2017). By comparing two groups of target words against two groups of attribute words, the approach quantifies bias in the embedding space.

### Implications for AI Applications 🌍
- **Real-World Impact:**  
  The observed biases could lead to perpetuation of gender stereotypes in applications that rely on these embeddings.
- **Need for Mitigation:**  
  Addressing such biases is crucial for ensuring ethical AI development and deployment.

### Limitations ⚠️
- **Small Target Sets:**  
  The analysis was performed on only 4 words per target set. A larger and more diverse set of words could provide more generalizable results.
- **Historical Data Bias:**  
  The GoogleNews embeddings were trained on older data, and therefore might reflect historical biases rather than current societal norms.
- **Generality:**  
  While the results are compelling, they pertain only to the specific target and attribute sets chosen for this evaluation.

*Emoji Tip: ⚠️🔍*

---

## 6. Conclusion 🎉

This report demonstrates how to compute a WEAT-like effect size for static word embeddings. With an effect size of **1.679**, the analysis reveals a strong bias in the GoogleNews embeddings:
- **Neutral Associations:**  
  Words in Target Set 1 (e.g., "doctor", "engineer") show nearly balanced associations with gendered attribute words.
- **Skewed Associations:**  
  Words in Target Set 2 (e.g., "nurse", "teacher") show a significant skew toward female attributes.

Even though the sample size is small, this methodology clearly illustrates how biases in word embeddings can be quantitatively assessed. Continued evaluation and the inclusion of more diverse word sets are essential to better understand and eventually mitigate these biases.


---
