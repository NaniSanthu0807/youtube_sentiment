# 📺 YouTube Comments Sentiment Analysis

## 📌 Project Overview
This project leverages **Machine Learning (ML)** and **Natural Language Processing (NLP)** to analyze user engagement on YouTube. By processing thousands of comments, the model classifies sentiments as **Positive, Negative, or Neutral**, providing content creators with an automated way to gauge audience reactions.  

The system can help in understanding audience mood, identifying popular trends, and monitoring engagement efficiently.

---

## 📊 Dataset Information
The dataset used is the **YouTube Comments Dataset**, containing thousands of real-world comments across various video categories.

**Source:** [Kaggle – YouTube Comments Dataset](YOUR_DATASET_LINK_HERE)  

**Key Features:**
- **Comment**: Raw text of the user comment.  
- **Likes**: Number of likes on the comment (useful for weighted sentiment).  
- **Sentiment (Target)**: Categorized sentiment label (Positive, Negative, Neutral).

---

## 🛠️ NLP & Preprocessing Pipeline
Processing social media text requires specialized steps to clean and structure the data for modeling:

1. **Data Cleaning**: Removing URLs, special characters, emojis, and HTML tags.  
2. **Tokenization**: Breaking sentences into individual words.  
3. **Stopword Removal**: Eliminating common words (e.g., "is", "the", "and") that carry little emotional weight.  
4. **Lemmatization**: Reducing words to their root form (e.g., "running" → "run").  
5. **Vectorization**: Converting text into numerical features using **TF-IDF** or **Bag of Words**.

---

## 📈 Machine Learning Models
Several classification algorithms were evaluated to determine which handles the nuances of internet slang and short comments most effectively:

| Model                       | Accuracy | Precision | Recall |
|-------------------------------|----------|-----------|--------|
| Logistic Regression           | 82%      | 0.81      | 0.80   |
| Multinomial Naive Bayes       | 79%      | 0.78      | 0.77   |
| Random Forest                 | 85%      | 0.84      | 0.83   |
| Support Vector Machine (SVM)  | 87%      | 0.86      | 0.86   |

> **Final Model:** SVM was selected for deployment due to its superior performance in high-dimensional text space.

---

## 💡 Key Features of the App
- **Single Comment Prediction**: Paste a YouTube comment to see its sentiment polarity.  
- **Batch Analysis**: Upload a CSV file of comments to get a summary report of overall audience sentiment.  
- **Word Cloud Visualization**: Generates visual representation of the most frequently used words in positive vs. negative comments.

---

## 🔗 Project Files
- `YouTube_Comments_Dataset.csv` – Preprocessed dataset.  
- `sentiment_analysis.ipynb` – Jupyter Notebook with preprocessing, modeling, and evaluation.  
- `requirements.txt` – Python dependencies.  

---

## 🛠️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
