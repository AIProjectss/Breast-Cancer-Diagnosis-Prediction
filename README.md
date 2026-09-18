# 🩺 Breast Cancer Diagnosis Predictor

**Capstone Mini-Project — Artificial Intelligence (Machine Learning & Deep Learning)**
NAVTTC | Prime Minister's Hunarmand Pakistan Program — "Skills for All"

**Author:** Syed Israr Ali
**Roll No:** 24

---

## 📌 Problem Statement

Early and accurate diagnosis of breast cancer significantly improves treatment outcomes. This project builds a machine learning model that classifies a breast tumor as **Malignant** or **Benign** using numeric measurements derived from a digitized image of a breast mass biopsy (Fine Needle Aspirate), and deploys it as an interactive web app.

## 📊 Dataset

- **Breast Cancer Wisconsin (Diagnostic) Dataset**
- Source: UCI Machine Learning Repository, loaded via `sklearn.datasets.load_breast_cancer()`
- 569 samples, 30 numeric features, binary target (malignant / benign)

## ⚙️ Approach

1. **EDA** — class distribution, correlation heatmap, feature distributions
2. **Preprocessing** — stratified train/test split, feature scaling (`StandardScaler`)
3. **Modeling** — Logistic Regression vs. Random Forest, compared on accuracy, precision, recall, F1, and ROC-AUC
4. **Evaluation** — confusion matrices, ROC curves, feature importance
5. **Deployment** — final Logistic Regression model served through a Streamlit web app

## 🗂️ Repository Structure

```
├── Assignment7_SyedIsrarAli.ipynb   # Full notebook: EDA, training, evaluation
├── app.py                           # Streamlit web app
├── model.pkl                        # Saved trained model
├── scaler.pkl                       # Saved fitted StandardScaler
├── feature_names.pkl                # Saved feature name list
├── requirements.txt                 # Python dependencies
├── SRS_Document.docx                # Software Requirements Specification
└── README.md
```

## 🚀 Run Locally

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

👉 **Streamlit App:** [add your Streamlit Cloud link here]

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (deployed) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes only. It is **not** a certified medical device and should never be used for actual clinical diagnosis.

## 🏷️ Tags

`#MachineLearning` `#DeepLearning` `#AI` `#Streamlit` `#NAVTTC` `#HunarmandPakistan` `#DataScience` `#Python`
