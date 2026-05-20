# 🔬 Breast Cancer Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.12-blue?logo=mlflow)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-green)

> **Group 2 — Machine Learning Project**  
> Binary classification: predict whether a breast tumour is **Malignant (Yes)** or **Benign (No)**  
> using Logistic Regression trained on the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## 📑 Table of Contents

1. [Project Description](#-project-description)  
2. [Tech Stack](#-tech-stack)  
3. [Project Structure](#-project-structure)  
4. [Setup & Installation](#-setup--installation)  
5. [Train the Model](#-train-the-model)  
6. [Run the Flask API](#-run-the-flask-api)  
7. [Run the Streamlit UI](#-run-the-streamlit-ui)  
8. [API Documentation](#-api-documentation)  
9. [MLflow Experiment Tracking](#-mlflow-experiment-tracking)  
10. [Deployment on Render](#-deployment-on-render)  
11. [Dataset Description](#-dataset-description)  
12. [Model Performance](#-model-performance)  
13. [Screenshots](#-screenshots)  
14. [GitHub Setup](#-github-setup)  
15. [License](#-license)  

---

## 📝 Project Description

This project implements a complete **MLOps pipeline** for breast cancer detection:

- **Model**: Logistic Regression (scikit-learn) achieving ~97% accuracy
- **Experiment Tracking**: MLflow logs parameters, metrics, confusion matrix, and model artifacts
- **REST API**: Flask API exposes a `/predict` endpoint accepting 30 numeric features
- **Frontend**: Streamlit app provides manual input and sample-data demo modes
- **Deployment**: Flask API hosted on Render; Streamlit can be deployed on Streamlit Cloud

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| ML Library | scikit-learn 1.4 |
| Experiment Tracking | MLflow 2.12 |
| REST API | Flask 3.0 + Flask-CORS |
| Frontend | Streamlit 1.34 |
| Deployment | Render (Flask API) |
| Process Manager | Gunicorn |
| Version Control | Git + GitHub |

---

## 📂 Project Structure

```
breast-cancer-prediction/
├── model/
│   ├── __init__.py
│   └── train.py          ← Training script with MLflow
├── api/
│   ├── __init__.py
│   └── app.py            ← Flask REST API
├── ui/
│   └── app.py            ← Streamlit frontend
├── model_artifacts/
│   ├── model.pkl         ← Saved LogisticRegression (generated after training)
│   ├── scaler.pkl        ← Saved StandardScaler   (generated after training)
│   └── confusion_matrix.png
├── mlruns/               ← Auto-created by MLflow
├── requirements.txt
├── render.yaml
├── Procfile
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python model/train.py
```

This will:
- Load the Breast Cancer Wisconsin dataset
- Train a Logistic Regression model
- Log parameters, metrics, confusion matrix to MLflow
- Save `model.pkl` and `scaler.pkl` to `model_artifacts/`

Expected output:
```
[INFO] Dataset loaded: 569 samples, 30 features
[INFO] Train: 455  |  Test: 114

======================================================
  MODEL PERFORMANCE METRICS
======================================================
  Accuracy  : 0.9737  (97.37%)
  Precision : 0.9726
  Recall    : 0.9863
  F1-Score  : 0.9794
  ROC-AUC   : 0.9967
======================================================
```

---

## 🚀 Run the Flask API

```bash
# Development
python api/app.py

# Production (gunicorn)
gunicorn api.app:app
```

The API will be available at `http://localhost:5000`.

---

## 🎨 Run the Streamlit UI

> Make sure the Flask API is running first.

```bash
streamlit run ui/app.py
```

The UI opens at `http://localhost:8501`.

---

## 📡 API Documentation

### `GET /`

Health check.

**Response:**
```json
{
  "status": "ok",
  "message": "Breast Cancer Prediction API",
  "version": "1.0.0"
}
```

---

### `POST /predict`

Predict Malignant / Benign given 30 feature values.

**Request:**
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
               0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
               0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
               0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Yes (Malignant)",
  "probability": {
    "malignant": 0.998712,
    "benign":    0.001288
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

**Error Response (wrong number of features):**
```json
{
  "error": "Expected 30 features, got 5.",
  "hint": "Provide exactly 30 numeric feature values."
}
```

---

## 📊 MLflow Experiment Tracking

After training, view all tracked experiments:

```bash
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser.

> ⚠️ If Flask is also running on port 5000, use:
> ```bash
> mlflow ui --port 5001
> ```

The MLflow UI shows:
- Run parameters: `C`, `max_iter`, `solver`, `test_size`, `random_state`
- Metrics: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`
- Artifacts: `model.pkl`, `scaler.pkl`, `confusion_matrix.png`

---

## ☁️ Deployment on Render

### Step 1 — Push to GitHub

```bash
git push origin main
```

### Step 2 — Connect Render

1. Go to [render.com](https://render.com) and sign in
2. Click **New → Web Service**
3. Connect your GitHub repository
4. Render detects `render.yaml` automatically

### Step 3 — Environment

Render will use:
- **Build Command**: `pip install -r requirements.txt && python model/train.py`
- **Start Command**: `gunicorn api.app:app`
- **Python**: 3.10

### Step 4 — Deploy

Click **Create Web Service**. Your API will be live at:
```
https://breast-cancer-prediction-api.onrender.com
```

Update `API_URL` in `ui/app.py` with your Render URL before deploying the UI.

---

## 📋 Dataset Description

**Source:** `sklearn.datasets.load_breast_cancer()`  
**Samples:** 569 | **Features:** 30 | **Classes:** Malignant (212), Benign (357)

The 30 features are computed from digitized images of fine needle aspirate (FNA) of breast masses.
Ten real-valued features are computed for each cell nucleus, then mean, standard error, and worst (largest) values are recorded:

| # | Feature Name |
|---|---|
| 1 | mean radius |
| 2 | mean texture |
| 3 | mean perimeter |
| 4 | mean area |
| 5 | mean smoothness |
| 6 | mean compactness |
| 7 | mean concavity |
| 8 | mean concave points |
| 9 | mean symmetry |
| 10 | mean fractal dimension |
| 11 | radius error |
| 12 | texture error |
| 13 | perimeter error |
| 14 | area error |
| 15 | smoothness error |
| 16 | compactness error |
| 17 | concavity error |
| 18 | concave points error |
| 19 | symmetry error |
| 20 | fractal dimension error |
| 21 | worst radius |
| 22 | worst texture |
| 23 | worst perimeter |
| 24 | worst area |
| 25 | worst smoothness |
| 26 | worst compactness |
| 27 | worst concavity |
| 28 | worst concave points |
| 29 | worst symmetry |
| 30 | worst fractal dimension |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~97.37% |
| Precision | ~97.26% |
| Recall | ~98.63% |
| F1-Score | ~97.94% |
| ROC-AUC | ~99.67% |

> Metrics are approximate and depend on random seed and data split. Run `python model/train.py` for exact values.

---

## 📸 Screenshots

> _Add screenshots here after running the app._

| Streamlit UI — Manual Input | Streamlit UI — Prediction Result |
|---|---|
| _(screenshot placeholder)_ | _(screenshot placeholder)_ |

| MLflow Experiment Dashboard | Confusion Matrix |
|---|---|
| _(screenshot placeholder)_ | _(screenshot placeholder)_ |

---

## 🔧 GitHub Setup

```bash
# 1. Initialize repo
git init
cd breast-cancer-prediction

# 2. Add .gitignore (already created — verify it excludes mlruns/, *.pkl, etc.)
cat .gitignore

# 3. Stage all files
git add .

# 4. First commit
git commit -m "feat: initial commit — breast cancer prediction MLOps project"

# 5. Add remote and push
git remote add origin https://github.com/<your-username>/breast-cancer-prediction.git
git branch -M main
git push -u origin main
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Group 2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built with ❤️ by Group 2 · Machine Learning Course*
