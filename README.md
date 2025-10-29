# NLP with Disaster Tweets (Kaggle)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red?style=for-the-badge&logo=keras)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

A complete machine learning project for the Kaggle competition ["NLP Getting Started"](https://www.kaggle.com/c/nlp-getting-started), which involves building a model to classify which tweets are about real disasters and which are not.

**This repository contains:**
* A full Exploratory Data Analysis (EDA).
* A baseline model (TF-IDF + Logistic Regression).
* An advanced model (Bidirectional LSTM) built with TensorFlow/Keras.
* A systematic comparison of different architectures (LSTM vs. GRU) and training techniques (EarlyStopping, fine-tuning).

**Final Model Validation F1-Score: `0.7753`**

---

## Project Overview

The goal of this project is to build a binary classification model that can read a tweet and determine if it is disaster-related. The core challenge is **contextual ambiguity**, as many words (like "fire" or "body") can be used in both a disaster and a non-disaster context.

This project solves this by:
1.  **Deeply analyzing** the text data to find patterns.
2.  **Establishing a baseline** F1-score of `0.75` using a classical TF-IDF model.
3.  **Building a sequential RNN** (Bidirectional LSTM) that learns the *context* of words using pre-trained GloVe embeddings.
4.  **Systematically tuning** the model to find the best-performing architecture and training strategy, which resulted in our final model.

---

## Methodology & Results

We ran a series of 6 experiments to compare architectures and training techniques. Our primary metric for success was the **Validation F1-Score**.

### Model Experiment Summary

| Model | Architecture | Key Technique | Validation F1-Score | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | TF-IDF + LogReg | N/A | **0.7500** (approx.) | Our score to beat. Ignores word order. |
| **Model 1** | Bi-LSTM | `Dropout(0.5)` | 0.7635 (Peak) | Beat the baseline but **overfit severely**. |
| **Model 2** | Bi-LSTM | `EarlyStopping (on val_loss)` | **0.7506** | Fixed overfitting, but optimized for the wrong metric. |
| **Model 3** | **Bi-LSTM** | **`EarlyStopping (on val_f1_score)`** | **0.7753** | **OUR BEST MODEL.** Captured the true peak F1-score. |
| **Model 4** | Bi-GRU | `EarlyStopping (on val_f1_score)` | **0.7724** | Great performance, but slightly worse than the LSTM. |
| **Model 6** | Bi-LSTM | `Fine-Tuning (trainable=True)` | **0.5691** | **Failed.** "Catastrophic forgetting" destroyed the weights. |

---

## Key Learnings & Conclusion

Our experiments provided several clear, actionable insights:

* **What Worked**
    * **Pre-trained Embeddings:** Using "frozen" (`trainable=False`) GloVe embeddings was the most powerful technique.
    * **`EarlyStopping`:** This was the single most critical technique for preventing overfitting.
    * **Monitoring the Right Metric:** The key to our best score was changing the `EarlyStopping` monitor from `val_loss` to `val_f1_score`.
    * **`Bidirectional-LSTM`:** This architecture was our champion, proving slightly better than `Bi-GRU` for this specific task.

* **What Did Not Work**
    * **Fine-Tuning Embeddings:** This technique (Model 6) was a clear failure. It led to "catastrophic forgetting" and destroyed the model's performance.
    * **Ignoring Overfitting:** Simply training for a fixed number of epochs (Model 1) was a failed strategy, as the model's best performance was at Epoch 1.

---

## How to Run

This project was built in a Google Colab notebook.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/csdscoursera/Week_4_DL_Project.git](https://github.com/csdscoursera/Week_4_DL_Project.git)
    cd Week_4_DL_Project
    ```

2.  **Install Dependencies:**
    The project uses the following major libraries.
    ```bash
    pip install tensorflow pandas numpy scikit-learn nltk wordcloud matplotlib seaborn
    ```

3.  **Get the Data:**
    * Download the data from the [Kaggle competition page](https://www.kaggle.com/c/nlp-getting-started/data).
    * You will need `train.csv`, `test.csv`, and `sample_submission.csv`.
    * (If using Colab, you can upload these or mount them from Google Drive as shown in the notebook).

4.  **Download GloVe Embeddings:**
    * The model requires the 100-dimension GloVe embeddings (`glove.6B.100d.txt`).
    * You can download them here: [GloVe Website](https://nlp.stanford.edu/projects/glove/) (glove.6B.zip).

5.  **Run the Notebook:**
    * Open and run the `.ipynb` notebook to see the full analysis, model training, and final submission generation.
