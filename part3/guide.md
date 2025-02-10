
# Contextual Model Bias Evaluation 🚀

This repository contains scripts designed to evaluate harmful associations and gender bias in contextual language models using masked language modeling (MLM). The aim is to measure how often male- and female-associated tokens are predicted at a masked position in sentences describing various professional roles. This analysis helps uncover potential gender biases in models like BERT.

---

## Folder Structure 📁

```
├── scripts/
│   ├── contextual_bias.py    # Core script to evaluate bias in contextual models
│   ├── download.py           # Utility script for downloading models/data (if needed)
│   └── static_bias.py        # Script for evaluating bias in static word embeddings
├── Contextual_Bias.md        # Documentation for contextual model bias evaluation
├── Static-Bias.md            # Documentation for static word embeddings bias evaluation
└── requirements.txt          # Python dependencies
```

---

## 1. Script: `contextual_bias.py` 🤖

This is the core script that evaluates gender bias in contextual language models. It computes the probability differences and ratios for male- vs. female-associated tokens in the context of different professional roles.

### Key Components:

### 1.1 Model Initialization & Setup 🔧

- **Loading Pretrained Model & Tokenizer:**
  - Uses `AutoModelForMaskedLM` and the corresponding tokenizer from Hugging Face’s `transformers` library.
  - Loads the model onto the specified device (`cuda` for GPU or `cpu` for CPU).
  
  *Example Code:*
  ```python
  from transformers import AutoModelForMaskedLM, AutoTokenizer
  model_name = "bert-base-uncased"
  model = AutoModelForMaskedLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  ```

### 1.2 Token Probability Calculation 🔢

- **Functionality:**
  - The script defines a function (e.g., `get_token_probabilities()`) to compute the probabilities of specific tokens (like "he" and "she") at the masked position.
  - It replaces `[MASK]` in a template sentence with the actual mask token provided by the tokenizer.
  - The model then outputs logits which are converted to probabilities using a softmax function.
  
  *Key Steps:*
  1. Replace `[MASK]` in the sentence template.
  2. Tokenize the sentence and identify the mask position.
  3. Run the model to get logits.
  4. Apply softmax and extract target token probabilities.
  
  *Emoji Tip: 📊*

### 1.3 Bias Evaluation Process 🔍

- **Evaluating Gender Bias:**
  - The function `evaluate_bias_for_roles()` loops through each professional role (e.g., "doctor", "nurse", "teacher") and for each role, it evaluates the bias using gendered token pairs such as ("he", "she") and ("his", "her").
  - For each role, the script calculates:
    - **Male Probability:** Likelihood of the male-associated token (e.g., "he").
    - **Female Probability:** Likelihood of the female-associated token (e.g., "she").
    - **Difference:** The result of (male probability - female probability).
    - **Ratio:** The result of (male probability / female probability). A ratio greater than 1 indicates a male bias, whereas a ratio less than 1 indicates a female bias.
  
  *Example Output:*
  ```
  Results:

  Role: doctor
    he vs she:
      Probabilities: 0.852 vs 0.148
      Difference: 0.704
      Ratio: 5.754

  Role: nurse
    he vs she:
      Probabilities: 0.394 vs 0.606
      Difference: -0.212
      Ratio: 0.651

  Role: engineer
    he vs she:
      Probabilities: 0.788 vs 0.212
      Difference: 0.576
      Ratio: 3.718
  ```
  - For the **doctor** role, a high male probability indicates a strong male bias.
  - For the **nurse** role, a higher female probability indicates a female bias.
  
  *Emoji Tip: 🤔📉📈*

### 1.4 Script Arguments & Running the Script 🏃‍♂️

- **Arguments:**
  - `--model_name`: Pretrained model name (default: `bert-base-uncased`).
  - `--device`: Device for inference (`cpu` or `cuda`).
  - `--batch_size`: Batch size for processing tokens (default: 32).

- **Example Command:**
  ```bash
  python contextual_bias.py --model_name bert-base-uncased --device cuda --batch_size 32
  ```

---

## 2. Script: `requirements.txt` 📦

The `requirements.txt` file lists the dependencies needed to run the scripts. The key dependencies include:

- **torch:** For model loading and inference.
- **transformers:** For accessing pretrained language models.
- **numpy:** For numerical operations.
- **tqdm:** For displaying progress bars during evaluation.

*To install dependencies, run:*
```bash
pip install -r requirements.txt
```

*Emoji Tip: 🛠️📥*

---

## 3. How It Works: Under the Hood 🧠

### Masked Language Modeling (MLM):
- **Concept:**
  - The model is trained to predict a missing word in a sentence where a word is replaced by a `[MASK]` token.
  - In our evaluation, we use a sentence template (e.g., “The doctor said that [MASK] is very busy.”) and test how the model fills in the mask with gendered tokens.

### Evaluating Gendered Associations:
- **Gendered Tokens:**
  - The script evaluates pairs of gendered tokens (e.g., "he" vs. "she", "his" vs. "her") to quantify bias.
- **Batch Processing:**
  - For efficiency, multiple tokens are processed in batches, reducing computation time and improving performance.

*Emoji Tip: 🔄🤖*

---

## 4. Conclusion & Next Steps 🎉

This repository provides a robust framework for evaluating gender bias in contextual language models like BERT. The script measures how likely the model is to associate different professional roles with male or female tokens. Key outcomes include:

- **Probability Difference:** Indicates which gender the model favors.
- **Probability Ratio:** Quantifies the strength of the bias.
  
**Example Insights:**
- A high probability for "he" in the context of "doctor" may indicate a male bias.
- A higher probability for "she" in the context of "nurse" indicates a female bias.

**Future Directions:**
- Extend the evaluation to include additional roles and other forms of bias.
- Experiment with different sentence templates to test the robustness of the bias evaluation.
- Integrate these evaluations into bias mitigation strategies for fairer AI systems.

*Emoji Tip: 🌟🤝*

---

## Final Note

This README explains the workflow, the scripts involved, and how to use them to evaluate bias in contextual models. Feel free to modify or extend the scripts to explore other dimensions of bias in your models.

Happy Evaluating! 😃🚀
```
