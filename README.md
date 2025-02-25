# 📝 Text Classification Report: SMS Spam Collection and 20 Newsgroups Datasets


## 🌟 1. Introduction

This report presents an implementation of **text classification** on two distinct datasets: the *SMS Spam Collection* dataset and the *20 Newsgroups* dataset. The objective is to preprocess textual data, extract features using **TF-IDF** and **Word2Vec**, and evaluate the performance of multiple classification models, including ***Naive Bayes***, ***Support Vector Machines (SVM)***, ***Logistic Regression***, and a ***Neural Network*** built with TensorFlow. The models are assessed using standard metrics such as *accuracy*, *precision*, *recall*, and *F1-score*, with results visualized for comparison.

- **SMS Spam Collection**: Binary-labeled text messages (*spam* or *ham*).
- **20 Newsgroups**: Multi-class dataset of news articles across 20 categories.

This analysis highlights the effectiveness of different approaches for **spam detection** (binary classification) and **news categorization** (multi-class classification).

---

## 🛠️ 2. Methodology

### 2.1 Datasets
- **SMS Spam Collection**: Loaded from `spam.csv`, containing **5,572 messages** labeled as `ham` (0) or `spam` (1).
- **20 Newsgroups**: Fetched via `sklearn.datasets.fetch_20newsgroups`, containing **18,846 documents** across 20 categories, with headers, footers, and quotes removed.

### 2.2 Data Preprocessing
A preprocessing function (`preprocess_text`) was applied to both datasets:
- Converted text to *lowercase*.
- Removed special characters and numbers using regex (`re.sub`).
- Tokenized text with NLTK’s `word_tokenize`.
- Removed stopwords and lemmatized tokens using `WordNetLemmatizer`.
- Joined tokens into a cleaned string (`processed_text` column).

### 2.3 Dataset Splitting
Datasets were split into *training (60%)*, *validation (20%)*, and *test (20%)* sets using `prepare_dataset` with stratification:
- **SMS**: Train: *3,343*, Validation: *1,114*, Test: *1,115*.
- **Newsgroups**: Train: *11,307*, Validation: *3,769*, Test: *3,770*.

### 2.4 Feature Extraction
Two methods were used:
1. **TF-IDF Vectorization**:
   - *SMS*: `TfidfVectorizer` with 5,000 max features → `(3,343, 5,000)`.
   - *Newsgroups*: `TfidfVectorizer` with 10,000 max features → `(11,307, 10,000)`.
2. **Word2Vec Embeddings**:
   - Custom Word2Vec models trained on training corpus (`vector_size=100, window=5, min_count=1`).
   - Sentence embeddings as mean of word vectors → `(3,343, 100)` for SMS, `(11,307, 100)` for Newsgroups.

### 2.5 Models
Four models were implemented:
1. **Naive Bayes**: `MultinomialNB` (TF-IDF features).
2. **SVM**: `SVC` (TF-IDF features).
3. **Logistic Regression**: `LogisticRegression` (TF-IDF features).
4. **Neural Network**: TensorFlow `Sequential` model:
   - `Dense(256, ReLU)` → `Dropout(0.4)`.
   - `Dense(128, ReLU)` → `Dropout(0.3)`.
   - `Dense(64, ReLU)` → `Dropout(0.2)`.
   - Output: `Dense(2, sigmoid)` (SMS); `Dense(20, softmax)` (Newsgroups).
   - Trained for *100 epochs* with *Adam* optimizer.

### 2.6 Evaluation Metrics
A custom `evaluate_model` function computed:
- *Accuracy*, *Precision*, *Recall*, *F1-Score* (weighted for multi-class).
- *Sensitivity* and *Specificity* (binary classification).
- *Classification Report* and *Confusion Matrix*.
- *ROC-AUC* (binary classification).

---

## 📊 3. Results

### 3.1 SMS Spam Detection
Performance on the SMS test set (*1,115 samples*):

| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------|--------------|---------------|------------|--------------|
| Naive Bayes          | 0.9614       | **0.9908**    | 0.7200     | 0.8340       |
| **SVM**              | **0.9839**   | 0.9714        | **0.9670** | **0.9379**   |
| Logistic Regression  | 0.9534       | 0.9712        | 0.6733     | 0.7953       |
| Neural Network       | 0.9193       | 0.7000        | 0.7000     | 0.7000       |

- **Best Model**: *SVM* with highest accuracy (0.9839) and balanced F1-score (0.9379).
- **Observations**: Naive Bayes led in precision; SVM excelled overall. Neural Network underperformed.

### 3.2 20 Newsgroups Classification
Performance on the Newsgroups test set (*3,770 samples*):

| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------|--------------|---------------|------------|--------------|
| Naive Bayes          | 0.7037       | **0.7233**    | 0.7037     | 0.6911       |
| SVM                  | 0.6997       | 0.7122        | 0.6997     | 0.7008       |
| **Logistic Regression** | **0.7090** | 0.7159        | **0.7090** | **0.7065**   |
| Neural Network       | 0.4459       | 0.4763        | 0.4459     | 0.4349       |

- **Best Model**: *Logistic Regression* with highest accuracy (0.7090), edging Naive Bayes by 0.0053.
- **Observations**: Neural Network lagged significantly (0.4459); traditional models performed consistently (~0.70).

### 3.3 Neural Network Training (Newsgroups)
- *Training Accuracy*: 0.1271 (Epoch 1) → 0.4550 (Epoch 100).
- *Validation Accuracy*: Peaked at 0.4582 (Epoch 89), final 0.4460.
- *Loss*: Stabilized at ~1.65–1.68, suggesting limited generalization.

---

## 🎨 4. Visualizations
- **Training History (Newsgroups NN)**: Accuracy/loss plots showed convergence; validation plateaued after ~50 epochs.
- **Model Comparison Bar Charts**:
  - *SMS*: SVM dominated.
  - *Newsgroups*: Logistic Regression and Naive Bayes led; Neural Network trailed.

---

## 🏁 5. Conclusion

This analysis successfully implemented text classification on the *SMS Spam Collection* and *20 Newsgroups* datasets. Key findings:

- **SMS Spam Detection**: *SVM* was most accurate (0.9839), excelling with TF-IDF features.
- **20 Newsgroups Classification**: *Logistic Regression* led (0.7090), narrowly beating Naive Bayes (0.7037).
- **Neural Network**: Underperformed (0.4459 for Newsgroups), possibly due to tuning or complexity.

### 💡 Recommendations
- Tune Neural Network hyperparameters or use pre-trained embeddings (*e.g., GloVe*).
- Explore ensemble methods for model synergy.
- Increase Word2Vec `vector_size` for richer embeddings.

This study showcases the strengths of traditional ML (*SVM*, *Logistic Regression*) versus deep learning for text classification.

---
