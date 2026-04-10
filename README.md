
# 🧠 FakeNews_Spam_Detection

### 🛡️ *Team FakeBusters* — Empowering truth through intelligent detection 🧩

---

## 📁 Project Structure

```
FakeNews_Spam_Detection/
│
├── data/               # raw & processed datasets (ignored in git)
├── notebooks/          # EDA, preprocessing, and experiment notebooks
├── src/                # source scripts (e.g., preprocess.py, train.py)
├── reports/            # dataset summary, sample rows, slides
├── requirements.txt    # Python dependencies
└── .gitignore          # ignored files and folders
```

---

## ⚙️ Setup Instructions (for Teammates)

### 🧩 1. Clone the Repository

```bash
git clone https://github.com/ajayram231210006/FakeNews_Spam_Detection.git
cd FakeNews_Spam_Detection
```

---

### 📂 2. Get the Data

Raw and processed data are **not included** in the repo.
Download the dataset and place it in:

```
data/raw/
```

#### 🔗 Data Sharing

Processed data (cleaned CSVs) are available on Google Drive:
[👉 Download Here](https://drive.google.com/file/d/1jIIusWgOHIVbW30oRrbHZ0e29gZ-fkah/view?usp=sharing)

---

### 💻 3. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 🧼 4. Run Preprocessing

```bash
python src\preprocess.py
```

---

## 👥 Team FakeBusters

| **Member**                 | **GitHub**                                                   | **Roles & Responsibilities**                                                            |
| -------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| 🧠 **Ajayram (Member A)**  | [@ajayram231210006](https://github.com/ajayram231210006)     | Data collection, preprocessing, exploratory data analysis (EDA), data handling pipeline |
| ⚙️ **Aniket (Member B)**   | [@aniketkumarsingh-9](https://github.com/aniketkumarsingh-9) | Model training & evaluation, feature engineering, model optimization                    |
| 💻 **Abhishek (Member C)** | [@abhi231210003](https://github.com/abhi231210003)           | Frontend integration, report generation, documentation, presentation preparation        |

---

## 🚀 Tech Stack

* **Languages:** Python 🐍
* **Libraries:** scikit-learn, pandas, numpy, nltk, matplotlib
* **Techniques:** NLP preprocessing, TF-IDF vectorization, Logistic Regression, Naive Bayes
* **Tools:** Jupyter Notebook, Git, VS Code

---

## 📈 Project Overview

This project aims to classify textual content as **Fake News** or **Spam/Ham** using **Machine Learning** and **Natural Language Processing (NLP)** techniques.

**Key objectives:**

* Clean and preprocess raw text data
* Extract features using vectorization methods (TF-IDF, Bag of Words)
* Train and evaluate ML models to achieve high accuracy
* Visualize and compare model performance

---

## 📫 Contact

For queries, feedback, or collaboration opportunities, connect with us via our GitHub profiles above.

---

✨ *Team FakeBusters — Empowering truth through intelligent detection!* ✨
