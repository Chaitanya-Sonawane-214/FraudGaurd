<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1.7+-orange?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-2.0+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Ollama-LLaMA_3.2-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

# 🛡️ FraudGuard — AI-Powered Credit Card Fraud Detection System

> A full-stack, production-ready fraud detection system combining **XGBoost machine learning**, **SHAP explainability**, **MLflow experiment tracking**, **FastAPI** backend, and a stunning **glassmorphism UI** — powered with AI-driven analysis through **Ollama (LLaMA 3.2)**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Model Training Pipeline](#-model-training-pipeline)
- [API Endpoints](#-api-endpoints)
- [Frontend Features](#-frontend-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**FraudGuard** is an end-to-end credit card fraud detection system designed to identify fraudulent transactions in real time. Built on a dataset of **284,807 real-world European credit card transactions** (with only **0.17% fraud rate**), the system uses advanced machine learning techniques to tackle extreme class imbalance and deliver high-accuracy fraud detection with explainable AI insights.

The system features:
- A **trained XGBoost classifier** with **97.71% ROC-AUC** and **88.7% fraud recall**
- **Hybrid scoring** — combining ML probability (60%) with a rule-based risk score (40%)
- **AI-powered explanations** via Ollama's LLaMA 3.2 model for every prediction
- A **premium, production-grade frontend** with real-time analysis, animated visualizations, and scan history

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **XGBoost Model** | Gradient boosted classifier trained on SMOTE-balanced data for high recall on fraud |
| 📊 **SHAP Explainability** | Global & local feature importance analysis to understand model decisions |
| 📈 **MLflow Tracking** | Full experiment tracking with metrics, parameters, and artifact logging |
| ⚡ **FastAPI Backend** | High-performance async REST API for prediction and AI explanation |
| 🤖 **AI Explanations** | Ollama (LLaMA 3.2) generates natural language fraud analysis for each transaction |
| 🎨 **Glassmorphism UI** | Stunning dark-mode frontend with particle backgrounds, animated orbs, and gauge charts |
| 🔄 **Hybrid Scoring** | Combines ML probability with a domain-driven risk scoring engine |
| 📱 **Responsive Design** | Fully responsive — works on desktop, tablet, and mobile |
| 📜 **Scan History** | In-session transaction scan history with verdict, probability, and risk level |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (HTML/CSS/JS)                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Hero    │  │Transaction│  │ Results  │  │   Scan       │   │
│  │  Section │  │  Scanner  │  │ Dashboard│  │   History    │   │
│  └──────────┘  └─────┬─────┘  └────┬─────┘  └──────────────┘   │
│                      │              │                            │
└──────────────────────┼──────────────┼────────────────────────────┘
                       │              │
              ┌────────▼──────────────▼────────┐
              │      FastAPI Backend Server     │
              │   (uvicorn main:app --reload)   │
              │                                 │
              │  ┌──────────┐  ┌─────────────┐  │
              │  │ /predict │  │  /explain   │  │
              │  │ endpoint │  │  endpoint   │  │
              │  └────┬─────┘  └──────┬──────┘  │
              │       │               │         │
              │  ┌────▼─────┐  ┌──────▼──────┐  │
              │  │ XGBoost  │  │   Ollama    │  │
              │  │ Model    │  │ (LLaMA 3.2)│  │
              │  │(model.pkl)│  │localhost:   │  │
              │  │          │  │  11434      │  │
              │  └──────────┘  └─────────────┘  │
              └────────────────────────────────┘
                       │
              ┌────────▼───────────────────────┐
              │     MLflow Tracking Server      │
              │  (sqlite:///mlflow.db)          │
              └────────────────────────────────┘
```

### Data Flow

```
User Input (12 features)
        │
        ▼
   Risk Score Calculation ──────────────┐
        │                               │
        ▼                               │
   Find Closest Transaction             │
   in Training Data                     │
        │                               │
        ▼                               │
   XGBoost Prediction                   │
   (probability + class)                │
        │                               │
        ▼                               ▼
   Hybrid Score = (ML × 0.6) + (Risk × 0.4)
        │
        ▼
   Final Verdict: FRAUD / LEGIT
        │
        ▼
   Ollama AI Explanation (async)
        │
        ▼
   Response to Frontend
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **Python 3.x** | Core programming language |
| **FastAPI** | High-performance async web framework |
| **Uvicorn** | ASGI server for FastAPI |
| **XGBoost** | Gradient boosted decision tree classifier |
| **Scikit-learn** | Data preprocessing, model evaluation, StandardScaler |
| **Imbalanced-learn (SMOTE)** | Synthetic Minority Oversampling for class balancing |
| **SHAP** | Explainable AI — feature importance analysis |
| **MLflow** | Experiment tracking, model versioning, artifact logging |
| **Pandas / NumPy** | Data manipulation and numerical computing |
| **Matplotlib / Seaborn** | Data visualization for EDA and model evaluation |
| **Ollama (LLaMA 3.2)** | Local LLM for natural language fraud explanations |
| **HTTPX** | Async HTTP client for Ollama API calls |
| **Pickle** | Model serialization and deserialization |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5** | Semantic page structure |
| **CSS3** | Glassmorphism design, animations, responsive layout |
| **Vanilla JavaScript** | Dynamic UI, API calls, gauge rendering, particle effects |
| **Google Fonts** | Inter + Space Grotesk typography |
| **Canvas API** | Particle background animation and gauge chart rendering |

### Infrastructure
| Technology | Purpose |
|---|---|
| **SQLite** | MLflow backend store |
| **Ollama** | Local LLM inference server |
| **Virtual Environment (venv)** | Python dependency isolation |

---

## 📁 Project Structure

```
Credit Card Fraud Detection System/
│
├── backend/
│   ├── main.py                          # FastAPI application (predict + explain endpoints)
│   ├── CreditCardFraudDetectionModel.ipynb  # Jupyter notebook — full ML pipeline
│   ├── model.pkl                        # Trained XGBoost model (serialized)
│   ├── scaler.pkl                       # StandardScaler (serialized)
│   ├── creditcard.csv                   # Raw dataset (284,807 transactions)
│   ├── df_original.csv                  # Processed dataset with engineered features
│   ├── X_test.csv                       # Test features for closest-transaction matching
│   ├── y_test.csv                       # Test labels
│   ├── mlflow.db                        # MLflow experiment tracking database
│   ├── mlartifacts/                     # MLflow model artifacts
│   └── __pycache__/                     # Python bytecode cache
│
├── frontend/
│   └── index.html                       # Complete single-page application (HTML + CSS + JS)
│
├── mlartifacts/                         # Root-level MLflow artifacts
├── mlflow.db                            # Root-level MLflow database
├── venv/                                # Python virtual environment
├── Launching Requirements.txt           # Quick-start launch instructions
└── README.md                            # This file
```

---

## 🔄 Data Pipeline

### 1. Data Loading & EDA
- **Dataset**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Shape**: 284,807 rows × 31 columns
- **Features**: `Time`, `V1`–`V28` (PCA-transformed), `Amount`, `Class`
- **Class Distribution**: 284,315 legitimate (99.83%) / 492 fraudulent (0.17%)

### 2. Preprocessing
- **Feature Scaling**: `StandardScaler` applied to `Amount` → `Amount_Scaled` and `Time` → `Time_Scaled`
- **Original columns dropped** after scaling
- **Train/Test Split**: 80/20 with `stratify=y` and `random_state=42`
  - Train: 227,845 samples (394 fraud)
  - Test: 56,962 samples (98 fraud)

### 3. SMOTE Balancing
- Applied **Synthetic Minority Oversampling Technique (SMOTE)** to training data
- **Before SMOTE**: 227,451 legit / 394 fraud
- **After SMOTE**: 227,451 legit / 227,451 fraud (perfectly balanced)

### 4. Feature Engineering (Risk Scoring)
The backend implements a **domain-driven risk scoring system** using categorical risk maps:

| Feature | Categories & Risk Scores |
|---|---|
| Transaction Type | Online Shopping (0.8), International (1.0), POS/Swipe (0.3), ATM (0.5) |
| Location Risk | Low (0.1), Medium (0.5), High (0.9) |
| Device Type | Mobile (0.6), Desktop (0.4), ATM Machine (0.5), POS Terminal (0.3) |
| Merchant Category | Grocery (0.1), Electronics (0.5), Travel (0.6), Fuel (0.4), Luxury (0.7), Gaming/Crypto (1.0) |
| Card Type | Debit (0.3), Credit (0.5), Prepaid (0.8) |
| Distance | Same City (0.1), Different City (0.5), Different Country (0.9) |
| Frequency Today | 1-2 times (0.2), 3-5 times (0.6), 5+ times (1.0) |
| Last Transaction Time | < 5 min (1.0), 5-30 min (0.6), 30 min-2 hrs (0.3), 2+ hrs (0.1) |
| Weekend | Yes (0.6), No (0.3) |
| Card Age | New < 3 months (0.8), Regular 3-12 months (0.4), Old 1+ year (0.2) |

**Composite Risk Score** = Average of all 10 category scores (normalized to 0–1)

---

## 🤖 Model Training Pipeline

### XGBoost Configuration
```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=1,       # SMOTE already balanced
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
```

### Hybrid Prediction Formula
```
ML Probability    = model.predict_proba(features)[fraud_class]
Risk Score        = average(all_risk_category_scores)
Combined Score    = (ML_Probability × 0.6) + (Risk_Score × 0.4)
Final Verdict     = "FRAUD" if Combined_Score > 0.4 else "LEGIT"
```

### Closest Transaction Matching
The system uses an innovative **closest transaction matching** algorithm:
1. Filter training data by **amount** (±30% tolerance)
2. Filter by **hour** (±3 hour range)
3. If risk score ≥ 0.6, prefer **known fraud transactions** from filtered set
4. Use the closest match's PCA features for XGBoost prediction
5. Scale `Amount` and `Time` using training data statistics

---

## 🌐 API Endpoints

### `GET /`
Health check endpoint.
```json
{"message": "🏦 FraudGuard API is running!"}
```

### `POST /predict`
Analyze a transaction for fraud.

**Request Body:**
```json
{
  "amount": 2500,
  "hour": 2,
  "transaction_type": "Online Shopping",
  "location_risk": "High Risk",
  "device_type": "Mobile",
  "merchant_category": "Gaming/Crypto",
  "card_type": "Prepaid Card",
  "distance": "Different Country",
  "frequency_today": "5+ times",
  "last_txn_time": "< 5 minutes",
  "is_weekend": "Yes",
  "card_age": "New Card (< 3 months)"
}
```

**Response:**
```json
{
  "prediction": 1,
  "verdict": "FRAUD",
  "probability": 72.45,
  "risk_score": 0.81
}
```

### `POST /explain`
Get AI-powered natural language explanation for a prediction.

**Response (from Ollama LLaMA 3.2):**
```json
{
  "signal_explanation": "This transaction exhibits multiple high-risk signals...",
  "pattern_analysis": "The late-night timing combined with the high amount...",
  "precautions": [
    "Immediately freeze the card and contact your bank",
    "Enable two-factor authentication for all transactions",
    "Review all recent transactions for unauthorized activity",
    "Report the incident to your local cybercrime authority"
  ]
}
```

---

## 🎨 Frontend Features

- **Particle Canvas Background** — 80 animated particles with interconnecting lines
- **Animated 3D Orb** — Pulsing gradient orb with orbiting ring system and glowing dots
- **Glassmorphism Cards** — Frosted glass effect with `backdrop-filter: blur(12px)`
- **Transaction Scanner** — 12-input form with sliders, dropdowns, and shimmer-effect submit button
- **Real-time Results Dashboard**:
  - Color-coded verdict banner (green/red)
  - Probability, risk score, and threat level metrics
  - Animated gauge chart (Canvas API)
  - Signal matrix with HIGH/MEDIUM/LOW badges
  - AI pattern analysis and threat response tabs
  - Persistent scan history
- **Responsive Design** — Full mobile/tablet breakpoints at 768px and 480px

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed with `llama3.2` model pulled
- Git

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "Credit Card Fraud Detection System"
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap mlflow jupyter notebook fastapi uvicorn httpx
```

### Step 4: Pull LLaMA Model (for AI explanations)
```bash
ollama pull llama3.2
```

### Step 5: Run the Notebook (Optional — for retraining)
```bash
cd backend
jupyter notebook CreditCardFraudDetectionModel.ipynb
```

---

## ▶️ Usage

Launch all services in order:

### 1. Start MLflow Tracking UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Access at: `http://127.0.0.1:5000`

### 2. Start Ollama Server
```bash
ollama serve
```
Runs on: `http://127.0.0.1:11434`

### 3. Start Backend Server
```bash
cd backend
uvicorn main:app --reload
```
API at: `http://127.0.0.1:8000`

### 4. Open Frontend
```bash
cd frontend
start index.html    # Windows
open index.html     # macOS
xdg-open index.html # Linux
```

---

## 📊 Model Performance

| Metric | Legit (Class 0) | Fraud (Class 1) |
|---|---|---|
| **Precision** | 1.00 | 0.16 |
| **Recall** | 0.99 | 0.89 |
| **F1-Score** | 1.00 | 0.27 |
| **Support** | 56,864 | 98 |

| Overall Metric | Value |
|---|---|
| **Accuracy** | 99% |
| **ROC-AUC Score** | **0.9771** |
| **Macro Avg F1** | 0.63 |
| **Weighted Avg F1** | 0.99 |

> 📝 **Note**: The model is optimized for **high recall on fraud** (89%) — this means it catches the vast majority of fraudulent transactions, at the cost of some false positives. In fraud detection, missing a real fraud is far more costly than flagging a legitimate transaction.

### Top SHAP Features (Global Importance)
The most important features for fraud detection (via SHAP analysis):
1. `V14` — Strongest fraud signal
2. `V10` — Second most important
3. `V12` — Third most important
4. `V4`, `V17`, `V11`, `V3` — Contributing features
5. `Amount_Scaled`, `Time_Scaled` — Scaled transaction metadata

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud) — Credit Card Fraud Detection Dataset
- **XGBoost**: Tianqi Chen & Carlos Guestrin
- **SHAP**: Scott Lundberg
- **MLflow**: Databricks
- **Ollama**: Local LLM inference
- **FastAPI**: Sebastián Ramírez

---

<p align="center">
  <b>Built with ❤️ using XGBoost · SHAP · FastAPI · MLflow · Ollama</b><br>
  <i>© 2026 FraudGuard</i>
</p>
