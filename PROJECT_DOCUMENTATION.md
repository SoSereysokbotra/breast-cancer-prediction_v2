# 🔬 Breast Cancer Prediction System — Project Documentation

**Group 2 | Machine Learning Course**
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — `data.csv`
**Model:** Logistic Regression | **Framework:** scikit-learn, MLflow, Flask, Streamlit

---

## 1. Project Purpose

### 1.1 Overview

This project builds a complete, end-to-end **Machine Learning application** that predicts whether a breast tumour is **Malignant (cancerous)** or **Benign (non-cancerous)** based on 30 numeric measurements extracted from digitized images of Fine Needle Aspirate (FNA) biopsies.

The goal is not just to train a model — it is to deliver a **production-ready MLOps pipeline** that covers every stage of the machine learning lifecycle:

| Stage | Tool / Component |
|---|---|
| Data preparation & model training | `model/train.py` + scikit-learn |
| Experiment tracking & reproducibility | **MLflow** |
| Model serving via API | Flask (`api/app.py`) |
| User-facing prediction interface | Streamlit (`ui/app.py`) |
| Cloud deployment | Render + Gunicorn |
| Version control & collaboration | Git + GitHub |

### 1.2 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for effective treatment. Manual diagnosis from biopsy images is time-consuming and subject to human error. A machine learning model trained on validated medical data can serve as a **decision-support tool** to assist clinicians in flagging potential malignancies quickly.

### 1.3 What the System Does

1. A trained **Logistic Regression** model analyses 30 cell nucleus measurements.
2. It outputs a binary prediction:
   - **`1` → Malignant (Yes)** — tumour is likely cancerous
   - **`0` → Benign (No)** — tumour is likely non-cancerous
3. It also returns the **probability** of each class (e.g., 99.6% Malignant).
4. The prediction is accessible via a **REST API** or directly through the **Streamlit web interface**.

### 1.4 Label Encoding

The dataset (`data.csv`) uses the `diagnosis` column with values:

| CSV Value | Meaning | Encoded as |
|---|---|---|
| `M` | Malignant (cancerous) | **1** |
| `B` | Benign (non-cancerous) | **0** |

---

## 2. Why We Use MLflow

### 2.1 What is MLflow?

**MLflow** is an open-source platform for managing the complete Machine Learning lifecycle. It provides four key components:

| Component | Purpose |
|---|---|
| **MLflow Tracking** | Log and query experiments (parameters, metrics, artifacts) |
| **MLflow Projects** | Package ML code in a reusable, reproducible format |
| **MLflow Models** | Package models for deployment across multiple platforms |
| **MLflow Registry** | Central model store for versioning and staging |

In this project, we use **MLflow Tracking** and **MLflow Models**.

### 2.2 Why MLflow for This Project?

#### ✅ Experiment Reproducibility
Every time we train the model, MLflow records **exactly what parameters were used**, what data was fed in, and what results were achieved. This means any team member can reproduce the exact same experiment at any time.

#### ✅ Parameter Tracking
Without MLflow, if we change the regularisation strength (`C`) or solver and get a different accuracy, we might forget what settings produced the best result. MLflow stores every run automatically:

```
Logged Parameters:
  C             = 1.0
  max_iter      = 10000
  solver        = lbfgs
  test_size     = 0.2
  random_state  = 42
  dataset       = data.csv
```

#### ✅ Metric Tracking
MLflow captures all evaluation metrics in one place, making it easy to **compare runs side-by-side**:

```
Logged Metrics:
  accuracy   = 0.9649  (96.49%)
  precision  = 0.9750
  recall     = 0.9286
  f1_score   = 0.9512
  roc_auc    = 0.9960
```

#### ✅ Artifact Storage
MLflow automatically saves and versions:
- `model.pkl` — the trained Logistic Regression model
- `scaler.pkl` — the fitted StandardScaler
- `confusion_matrix.png` — visual performance report

#### ✅ Visual Dashboard
MLflow provides a **built-in web dashboard** where you can browse all runs, compare metrics across experiments, and download artifacts without writing any extra code.

#### ✅ Industry Standard
MLflow is used in real-world ML engineering at companies like Databricks, Microsoft, and many others. Using it in this project demonstrates professional MLOps practices.

### 2.3 How to View the MLflow Dashboard

> **All three services must be running before you open the browser.**

```powershell
# Navigate to the project folder first
cd "d:\Year2\Semester1\Machine Learning\Group 2\breast-cancer-prediction"

# Run MLflow UI (in a separate terminal)
.venv\Scripts\mlflow.exe ui --port 5001
```

Then open your browser and go to: **http://localhost:5001**

### 2.4 What You See in the MLflow Dashboard

```
Experiments
  └── breast-cancer-logistic-regression
        └── Run: logistic-regression-run
              ├── Parameters:  C, max_iter, solver, test_size, random_state
              ├── Metrics:     accuracy, precision, recall, f1_score, roc_auc
              └── Artifacts:
                    ├── artifacts/model.pkl
                    ├── artifacts/scaler.pkl
                    └── plots/confusion_matrix.png
```

---

## 3. How to Run the Full Project

### 3.1 Prerequisites

- Python 3.10 installed
- Project folder: `breast-cancer-prediction/`
- `data.csv` placed in the **parent folder** (`Group 2/data.csv`)

### 3.2 Step-by-Step Startup

Open **three separate terminal windows**, all pointed to the same folder:

```powershell
cd "d:\Year2\Semester1\Machine Learning\Group 2\breast-cancer-prediction"
```

**Terminal 1 — Train the model (run once)**
```powershell
.venv\Scripts\python.exe model/train.py
```

**Terminal 2 — Start the Flask API**
```powershell
.venv\Scripts\python.exe api/app.py
```

**Terminal 3 — Start the Streamlit UI**
```powershell
.venv\Scripts\streamlit.exe run ui/app.py
```

**Terminal 4 (optional) — View MLflow dashboard**
```powershell
.venv\Scripts\mlflow.exe ui --port 5001
```

### 3.3 Access Points

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Flask API | http://localhost:5000 |
| MLflow Dashboard | http://localhost:5001 |

---

## 4. Streamlit UI — How to Use It

### 4.1 Overview of the Interface

When you open **http://localhost:8501**, you will see:

```
┌─────────────────────────────────────────────────────┐
│  🔬 Breast Cancer Prediction App                    │
│  Logistic Regression · Breast Cancer Wisconsin      │
│                                                     │
│  [ ✏️ Manual Input ]  [ 🧪 Use Sample Data ]        │
│                                                     │
│  📊 Enter Feature Values                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Feature 1│ │ Feature 2│ │ Feature 3│            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ... (30 features total, 3 per row)                 │
│                                                     │
│  [ 🔮 Predict ]                                     │
│                                                     │
│  ⚕️ This tool is for educational purposes only.    │
└─────────────────────────────────────────────────────┘
```

### 4.2 Sidebar

The **left sidebar** contains:
- A description of the app and model
- The label encoding explanation (`M=1 Malignant`, `B=0 Benign`)
- A numbered list of all **30 feature names** for reference
- Group information

### 4.3 Input Mode Selection

At the top of the main area, there is a **radio button** to choose your input method:

#### Option A — `✏️ Manual Input`
- All 30 input fields are pre-filled with the **mean values** from `data.csv`.
- You can manually change any value by clicking on the number field and typing.
- Each field has a **min** and **max** boundary based on the real dataset range.
- Format: decimal numbers (e.g., `17.990000`)

> **When to use:** When you have real patient measurement data and want to get a prediction.

#### Option B — `🧪 Use Sample Data`
- A **dropdown** appears letting you choose between:
  - `🔴 Known Malignant Sample` — loads the first Malignant row from `data.csv`
  - `🟢 Known Benign Sample` — loads the first Benign row from `data.csv`
- All 30 fields are automatically filled with the chosen sample's values.
- An info box confirms which sample was loaded.

> **When to use:** For demonstration, testing, or verifying the model works correctly.

### 4.4 The 30 Input Fields

The 30 features are arranged in a **3-column grid** (10 rows). They represent measurements computed from digitized images of cell nuclei:

| # | Feature | Description |
|---|---|---|
| 1 | `radius_mean` | Mean of distances from center to perimeter |
| 2 | `texture_mean` | Standard deviation of grey-scale values |
| 3 | `perimeter_mean` | Mean perimeter of the nucleus |
| 4 | `area_mean` | Mean area of the nucleus |
| 5 | `smoothness_mean` | Mean local variation in radius lengths |
| 6 | `compactness_mean` | Mean of (perimeter² / area − 1.0) |
| 7 | `concavity_mean` | Mean severity of concave portions |
| 8 | `concave points_mean` | Mean number of concave portions |
| 9 | `symmetry_mean` | Mean symmetry of nucleus |
| 10 | `fractal_dimension_mean` | Mean "coastline approximation" − 1 |
| 11–20 | `*_se` | Standard error of each of the above 10 |
| 21–30 | `*_worst` | Largest (worst) value of each of the above 10 |

> **Tip:** Features 1–10 are most interpretable. Larger `radius_mean`, `area_mean`, and `concavity_mean` values tend to be associated with malignant tumours.

### 4.5 Making a Prediction

1. Select your **input mode** (Manual or Sample Data).
2. If using Sample Data, choose Malignant or Benign from the dropdown.
3. Review or adjust the **30 feature values** as needed.
4. Click the **`🔮 Predict`** button.
5. The app sends the data to the Flask API at `http://localhost:5000/predict`.
6. The result appears below the button.

### 4.6 Reading the Results

#### If Malignant is predicted:
```
┌────────────────────────────────────┐
│  🔴  Yes (Malignant)               │  ← Red banner
└────────────────────────────────────┘
⚠️ The model predicts this tumour is Malignant.
   Please consult a medical professional immediately.

📈 Prediction Probabilities
  Malignant (Yes)  ████████████████████  100%
  Benign    (No)   ▏                       0%

  🔴 Malignant Probability: 99.87%
  🟢 Benign Probability:     0.13%
```

#### If Benign is predicted:
```
┌────────────────────────────────────┐
│  🟢  No (Benign)                   │  ← Green banner
└────────────────────────────────────┘
✅ The model predicts this tumour is Benign.
   Regular check-ups are still recommended.

📈 Prediction Probabilities
  Malignant (Yes)  ██                    6.4%
  Benign    (No)   ██████████████████   93.6%

  🔴 Malignant Probability:  6.38%
  🟢 Benign Probability:    93.62%
```

### 4.7 Error Handling

| Situation | What you'll see |
|---|---|
| Flask API not running | 🚫 "Cannot connect to Flask API. Make sure `python api/app.py` is running." |
| API takes too long | ⏱️ "Request timed out." |
| Wrong number of features | ❌ API error message returned |

---

## 5. Flask API — Quick Reference

### Endpoint 1: Health Check

```http
GET http://localhost:5000/
```

**Response:**
```json
{
  "status": "ok",
  "message": "Breast Cancer Prediction API",
  "version": "1.0.0"
}
```

### Endpoint 2: Predict

```http
POST http://localhost:5000/predict
Content-Type: application/json

{
  "features": [17.99, 10.38, 122.8, 1001, 0.1184, ... (30 values)]
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Yes (Malignant)",
  "probability": {
    "malignant": 0.998712,
    "benign": 0.001288
  }
}
```

---

## 6. Disclaimer

> ⚕️ This system is built for **educational and academic purposes only**.
> It is **NOT** a certified medical device and must **NOT** be used for actual clinical diagnosis.
> Always consult a qualified healthcare professional for medical advice.

---

*Prepared by Group 2 — Machine Learning Course*
