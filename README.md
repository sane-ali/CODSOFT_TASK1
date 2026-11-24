# Task 1: Movie Genre Classification 🎬

This repository contains the solution for **Task 1: Movie Genre Classification** completed as part of the Machine Learning Virtual Internship at **CodSoft**.

---

## 📌 Project Overview
The goal of this project is to build a Machine Learning model capable of predicting the genre of a movie based on its plot summary/description. 

Techniques used:
- **Feature Extraction:** TF-IDF Vectorization (`ngram_range=(1, 2)`, `sublinear_tf=True`)
- **Classifiers Evaluated:**
  1. **Multinomial Naive Bayes**
  2. **Logistic Regression**
  3. **Linear Support Vector Classifier (Linear SVC)**

---

## 📁 Dataset Format
The model processes structured text data containing:
- `ID` ::: `TITLE` ::: `GENRE` ::: `DESCRIPTION`

---

## ⚙️ Workflow & Steps
1. **Data Preprocessing & Cleaning:** Text lowercasing, special characters/punctuation removal, and whitespace normalization.
2. **Feature Engineering:** Combining title and description text, vectorized via `TfidfVectorizer` (top 20,000 features).
3. **Model Training & Comparison:** Models trained on `train_data.txt` and evaluated on `test_data_solution.txt`.
4. **Visualizations Generated:**
   - `genre_distribution.png` — Distribution of movie genres in the training data.
   - `model_comparison.png` — Accuracy comparison across Naive Bayes, Logistic Regression, and Linear SVC.
   - `confusion_matrix.png` — Heatmap of actual vs. predicted genres for the best model.
5. **Custom Prediction:** Interactive function `predict_movie(title, description)` to predict genres for new movies along with Top 3 candidate genres.

---

## 📊 Key Results & Performance
- **Best Performing Model:** Linear Support Vector Classifier (Linear SVC)
- **Primary Metrics Tracked:** Accuracy, Precision, Recall, and Weighted/Macro F1-Scores.

---

## 💡 Key Takeaways & Conclusion
1. **Model Efficiency:** Linear SVC outperforms basic Naive Bayes and Logistic Regression for text classification on sparse TF-IDF vectors.
2. **Impact of Class Imbalance:** Highly populated genres (e.g., Drama, Documentary) achieve higher F1-scores, while rare genres benefit from hyperparameter tuning (`C` parameter adjustment / balanced class weights).
3. **Future Improvements:** Using pre-trained transformer embeddings (e.g., BERT, RoBERTa) can further capture semantic context in plot summaries beyond TF-IDF n-grams.

---

## 🚀 How to Run (Google Colab / Python)
1. Open `Task1_Movie_Genre.ipynb` in Google Colab.
2. Upload `train_data.txt` and `test_data_solution.txt`.
3. Run all cells sequentially to view evaluation reports and generated charts.

---
*Created for CodSoft Machine Learning Internship (#codsoft)*
