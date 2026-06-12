# 🎓 ISPS — Intelligent Student Performance System

> An agentic AI pipeline that predicts student academic risk in real time using FCM clustering, SVM classification, SHAP explainability, ARIMA drift detection, and a RAG-augmented ReAct decision engine.

---

## 📌 Overview

ISPS is a full-stack intelligent system built to identify students at academic risk — **before** they fail. It takes behavioral and demographic input, runs it through a multi-signal agentic pipeline, classifies the student as **HIGH / MEDIUM / LOW RISK**, generates a natural-language explanation, and dispatches a personalized intervention plan via email and Excel logging.

The system runs as a **FastAPI backend** with an interactive HTML dashboard frontend.

---

## 🧠 Pipeline Architecture

```
Student Input
     │
     ▼
┌─────────────────────┐
│   1. Triage Check   │  → Instant HIGH RISK if attendance < 60% or hours < 2
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. FCM Clustering  │  → Fuzzy membership scores (μ) across 3 clusters
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   3. SVM Predict    │  → Class probabilities + SVM confidence gap
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. SHAP Reliability│  → Cosine similarity of student SHAP vs cluster mean
└────────┬────────────┘       + SHAP re-check loop across clusters
         │
         ▼
┌─────────────────────┐
│  5. ARIMA Trend     │  → Trend multiplier per cluster (rising / declining)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Intelligent     │  → Weighted composite of 5 independent signals
│     Score (i_score) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  7. ReAct 7-Path Routing Engine     │
│                                     │
│  Path 0 → Pre-Scoring Triage        │
│  Path 1 → Confident Dispatch        │
│  Path 2 → Standard Monitoring       │
│  Path 3 → RAG-Guided Majority Vote  │
│  Path 4 → RAG Tie-Break             │
│  Path 5 → Model Uncertainty         │
│  Path 6 → RAG Zero-Evidence         │
│  Path 7 → Human Review Escalation   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  8. Action Plan +   │  → Personalized intervention steps
│     NL Explanation  │     + Auto-generated natural language report
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Email Alert  │  Excel Log  │  API  │
└─────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| ML Models | SVM (calibrated), FCM (Fuzzy C-Means) |
| Explainability | SHAP |
| Time-Series | ARIMA / SARIMA (via `pmdarima`, `statsmodels`) |
| RAG Index | KNN (cosine similarity) over SHAP vectors |
| Frontend | Vanilla HTML/CSS/JS dashboard |
| Logging | openpyxl (styled Excel) + in-memory audit log |
| Email | Gmail SMTP (HTML alerts) |
| Balancing | SMOTE (`imbalanced-learn`) |
| Data | `StudentPerformanceFactors.csv` (6,600+ student records) |

---

## 📁 Project Structure

```
ISPS/
├── main.py                          # FastAPI backend — full agentic pipeline
├── dashboard.html                   # Interactive frontend dashboard
├── isps_student_log.xlsx            # Auto-generated styled Excel log
├── StudentPerformanceFactors.csv    # Raw dataset
├── Project_Intelligent_Student_Performance.ipynb  # Training notebook
└── models/
    ├── svm_model__1_.pkl            # Trained calibrated SVM
    ├── scaler__1_.pkl               # StandardScaler
    ├── fcm_centers__1_.npy          # FCM cluster centroids
    ├── feature_cols.pkl             # Feature column order
    ├── shap_values.pkl              # Precomputed SHAP values
    ├── cluster_shap_means.pkl       # Per-cluster SHAP means
    ├── trend_multipliers.pkl        # ARIMA-derived trend multipliers
    └── master_intervention_db.pkl   # Action plan database
```

---

## 📊 Dataset

`StudentPerformanceFactors.csv` — 6,607 student records with 20 features:

**Numerical:** `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score`

**Categorical:** `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`

**Target:** `Exam_Score` → binned into `Low / Medium / High` risk categories

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install fastapi uvicorn scikit-learn imbalanced-learn shap pmdarima statsmodels scikit-fuzzy openpyxl scipy numpy pandas
```

### 2. Add Trained Models

Place all `.pkl` and `.npy` files in a `models/` directory at the project root (see structure above).

### 3. Configure Email (Optional)

In `main.py`, update the email section:

```python
EMAIL_ADDRESS    = "your_email@gmail.com"
EMAIL_PASSWORD   = "your_app_password"       # Gmail App Password
SUPERVISOR_EMAIL = "supervisor@example.com"
```

> Use a Gmail **App Password** (not your account password). Enable 2FA first, then generate one at myaccount.google.com → Security → App Passwords.

### 4. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the Dashboard

Navigate to `http://localhost:8000/dashboard`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System status |
| `POST` | `/predict` | Predict risk for a single student |
| `POST` | `/batch-evaluate` | Predict risk for multiple students |
| `GET` | `/dashboard` | Serve the HTML dashboard |
| `GET` | `/stats` | Overall prediction statistics |
| `GET` | `/cluster-health` | FCM cluster distribution and health |
| `GET` | `/drift-status` | Dual-signal drift detection (SVM gap + centroid shift) |
| `GET` | `/shap-reliability` | SHAP explainability metrics |
| `GET` | `/audit-log` | Recent prediction history |
| `GET` | `/live-feed` | Last N predictions for live feed |
| `GET` | `/agentic-audit` | Agentic property verification |
| `GET` | `/centroid-pulse` | Live FCM centroid shift tracking |
| `GET` | `/export-excel` | Download full student log as `.xlsx` |
| `POST` | `/outcome` | Log a student outcome (improved / stable / declined) |
| `GET` | `/outcome-stats` | Intervention recovery rate stats |
| `GET` | `/random-student` | Generate and predict a random student profile |

### Example Request — `/predict`

```json
POST /predict
{
  "student_id": "S12345",
  "hours_studied": 5,
  "attendance": 55,
  "previous_scores": 48,
  "sleep_hours": 5,
  "motivation_level": 0,
  "tutoring_sessions": 0,
  "parental_involvement": 0,
  "access_to_resources": 1,
  "extracurricular_activities": 0,
  "internet_access": 1,
  "family_income": 0,
  "teacher_quality": 1,
  "school_type": 0,
  "peer_influence": 0,
  "physical_activity": 1,
  "learning_disabilities": 0,
  "parental_education_level": 0,
  "distance_from_home": 2,
  "gender": 0,
  "contact_email": "student@example.com"
}
```

### Example Response

```json
{
  "student_id": "S12345",
  "risk_level": "HIGH RISK",
  "intelligent_score": 0.92,
  "react_path": 0,
  "react_path_name": "Pre-Scoring Triage",
  "dominant_feature": "attendance",
  "nl_explanation": "Student classified as HIGH RISK via pre-scoring triage. Primary driver: attendance. Cluster trend is declining (multiplier=0.80). SHAP cosine similarity: 0.70. Confidence: 92%.",
  "action_plan": {
    "title": "Attendance Recovery Plan",
    "target": "School Advisor",
    "intensity": "HIGH",
    "priority": "IMMEDIATE",
    "timing": "within 72 hours",
    "steps": ["..."]
  },
  "requires_human_review": false
}
```

---

## 🧩 Agentic Properties

ISPS is designed as a **fully agentic system** with 6 verifiable properties:

| Property | Status | Description |
|---|---|---|
| Observes Continuously | ✅ Active | Tracks SVM gap and centroid shift per request |
| Reasons Before Acting | ✅ Active | 7-path ReAct loop runs on every prediction |
| Acts with Explanation | ✅ Active | NL explanation + SHAP dominant feature per student |
| Adapts to Outcomes | ✅ Active | Outcome log tracks recovery rate across interventions |
| Detects Own Degradation | ✅ Active | Dual-signal drift detection (SVM + centroid) |
| Retrains Autonomously | ⏳ Standby | Outer loop ready; gated on ≥2% accuracy improvement |

---

## 📈 Intelligent Score Formula

```
i_score = 0.25 × severity
        + 0.20 × svm_gap
        + 0.25 × fcm_membership (μ)
        + 0.15 × arima_trend_multiplier
        + 0.15 × shap_cosine_similarity
```

---

## 📬 Email Alerts

The system auto-sends a styled HTML email on every prediction to the supervisor (and optionally the student's contact email). Alert color, urgency messaging, and action steps are dynamically generated based on risk level.

---

## 📝 Excel Logging

Every prediction is automatically appended to `isps_student_log.xlsx` with color-coded rows (red = HIGH, orange = MEDIUM, green = LOW) and 27 tracked fields including all signal values, ReAct path, ARIMA vote, and the full NL explanation.

---

## 📓 Training Notebook

`Project_Intelligent_Student_Performance.ipynb` covers the full ML pipeline:
- Data cleaning and EDA
- Feature engineering and label encoding
- FCM clustering with fuzzy membership
- SVM training with SMOTE balancing
- SHAP value computation
- ARIMA trend modeling
- Model serialization

---

## 🔒 Notes

- The `models/` directory is not included in this repo. Train using the notebook or request pre-trained artifacts separately.
- Email credentials in `main.py` should be moved to environment variables before any production deployment.
- The system is designed for **educational/research** use.

---

## 👩‍💻 Author

Built as a graduation project — Faculty of Computers and Information Systems, Egyptian Chinese University.
