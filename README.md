# Emotion Detection from Arabic Tweets

Classifying Arabic tweets into 8 emotions, comparing classical machine learning models with deep learning models across three text representations.

**Best result:** SVM with TF-IDF reached a **Macro F1 of 0.6368** (64% accuracy) on a held-out test set of 2,013 tweets.

## Overview

The project compares how well different models and text representations detect emotion in Arabic text. Five classical ML models, a feed-forward neural network (FNN), and a bidirectional LSTM were trained and evaluated on the same stratified split.

**Emotion classes (8):** anger, fear, joy, love, none, sadness, surprise, sympathy.

## Dataset

- **Emotional-Tone Arabic dataset**: Arabic tweets labeled with 8 emotion categories.
- **Split:** 80% train / 20% test, stratified to keep the class distribution.
- **Test set size:** 2,013 samples.

## Pipeline

**1. Preprocessing**
- Removal of Arabic diacritics (tashkeel)
- Removal of punctuation and digits
- Removal of Arabic stopwords (NLTK)

**2. Text representations**

| Representation | Settings |
|---|---|
| Bag of Words | CountVectorizer, `max_features=5000` |
| TF-IDF | TfidfVectorizer, `max_features=5000`, `ngram_range=(1,2)` |
| Word2Vec | Trained on the dataset, `vector_size=100`, `window=5`, `epochs=10` |

**3. Models**
- Classical ML: SVM, Naive Bayes, Random Forest, Decision Tree, AdaBoost (multiple hyperparameter configurations each)
- Deep learning: Feed-Forward Neural Network (FNN), Bidirectional LSTM (3 configurations each, 10 epochs)

## Results

| Model | Best configuration | Macro F1 |
|---|---|---|
| **SVM** | TF-IDF, linear kernel, C=1 | **0.6368** |
| Bi-LSTM | units=64, embedding=128, lr=0.001 | 0.6165 |
| FNN | lr=0.001, dropout=0.5, units=256 | 0.6025 |

Other classical models (best Macro F1): Naive Bayes 0.5996, Random Forest 0.4874, Decision Tree 0.4457, AdaBoost 0.3721.

**Per-class performance (best model, SVM + TF-IDF):**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| Fear | 0.99 | 0.84 | 0.91 |
| Sympathy | 0.91 | 0.84 | 0.88 |
| None | 0.62 | 0.84 | 0.71 |
| Love | 0.70 | 0.66 | 0.68 |
| Anger | 0.55 | 0.66 | 0.60 |
| Joy | 0.45 | 0.47 | 0.46 |
| Surprise | 0.56 | 0.35 | 0.44 |
| Sadness | 0.46 | 0.39 | 0.42 |

## Key findings

- **SVM with TF-IDF outperformed both deep learning models.** With a limited amount of training data, the linear SVM generalized better than the FNN and Bi-LSTM.
- **Fear and Sympathy are the easiest emotions** to detect (F1 around 0.88 to 0.91), likely because of distinctive vocabulary.
- **Joy, Sadness, and Surprise are the hardest** (F1 around 0.42 to 0.46), suggesting overlap between these emotions in Arabic text.
- TF-IDF with unigrams and bigrams gave the strongest representation across models.

## Future work

- Fine-tune pre-trained Arabic transformer models such as AraBERT or CAMeL.
- Use pre-trained Arabic embeddings instead of training Word2Vec on a small dataset.
- Collect more data for the weaker classes (joy, sadness, surprise).

## Repository contents

| File | Description |
|---|---|
| `src.ipynb` | Full code: preprocessing, feature extraction, training, and evaluation |
| `Emotion_Detection_Report.docx` | Complete project report with all experiments and results |

## How to run

1. Open `src.ipynb` in Jupyter Notebook or Google Colab.
2. Install the libraries imported in the first cells of the notebook.
3. Run the cells in order.

## Author

**Fadia Mahmoud**, Data Science and Artificial Intelligence, The Hashemite University
[LinkedIn](https://www.linkedin.com/in/fadia-mahmoud-970061319) | [GitHub](https://github.com/fadia2004)
