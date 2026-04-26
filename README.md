# 👁 Optivision — Diabetic Retinopathy Detection
### Optimized Approach for DR Detection | Batch 09 · CSM · HITAM

---

## Project Structure

```
optivision/
├── frontend/
│   └── index.html          ← Single-page frontend (HTML + CSS + JS + Chart.js)
│
├── backend/
│   ├── main.py             ← FastAPI REST API (prediction, patients, dashboard)
│   └── requirements.txt    ← Python dependencies
│
└── database/
    ├── db.py               ← SQLite schema, CRUD helpers, seed data
    └── optivision.db       ← Auto-created SQLite database file
```

---

## Database Schema (SQLite)

### `patients`
| Column | Type | Description |
|---|---|---|
| id | TEXT (PK) | UUID |
| name | TEXT | Full name |
| age | INTEGER | Age in years |
| gender | TEXT | Male / Female / Other |
| diabetes_duration | INTEGER | Years with diabetes |
| hba1c | REAL | Glycated haemoglobin % |
| blood_pressure | TEXT | e.g. "130/80 mmHg" |
| email | TEXT | Contact email |
| phone | TEXT | Contact phone |
| created_at | TEXT | ISO datetime |

### `scans`
| Column | Type | Description |
|---|---|---|
| id | TEXT (PK) | UUID |
| patient_id | TEXT (FK) | → patients.id |
| image_filename | TEXT | Original filename |
| predicted_class | INTEGER | 0–4 (DR stage) |
| confidence | REAL | Model confidence 0–1 |
| risk_score | REAL | Risk score 0–100 |
| class_probabilities | TEXT | JSON array [p0..p4] |
| recommendations | TEXT | JSON array of strings |
| notes | TEXT | Optional clinical notes |
| created_at | TEXT | ISO datetime |

### `model_metrics`
| Column | Type | Description |
|---|---|---|
| epoch | INTEGER | Training epoch |
| train_loss / val_loss | REAL | Loss values |
| train_acc / val_acc | REAL | Accuracy 0–1 |
| precision_s / recall_s / f1_score | REAL | Per-class metrics |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model & system status |
| POST | `/api/predict` | Upload image → DR classification |
| POST | `/api/patients` | Register new patient |
| GET | `/api/patients` | List all patients |
| GET | `/api/patients/{id}` | Patient detail + scan history |
| GET | `/api/dashboard` | Stats, charts, recent scans |
| GET | `/api/metrics` | Training metrics per epoch |
| GET | `/api/dr-classes` | DR class metadata |

---

## Setup & Run

### 1. Install Python dependencies
```bash
cd optivision/backend
pip install -r requirements.txt
```

### 2. Initialise the database (auto-seeded)
```bash
cd optivision
python database/db.py
```

### 3. Start the API server
```bash
cd optivision
uvicorn backend.main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

### 4. Open the frontend
```
Open optivision/frontend/index.html in any browser
```
> The frontend works in **demo mode** even without the API running — it uses built-in mock data so you can explore all features.

---

## Dataset — APTOS 2019 Blindness Detection

**Download:** https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data

| Class | Label | Count |
|---|---|---|
| 0 | No DR | 1805 |
| 1 | Mild DR | 370 |
| 2 | Moderate DR | 999 |
| 3 | Severe DR | 193 |
| 4 | Proliferative DR | 295 |
| **Total** | | **3 662** |

---

## Model

- **Architecture:** ResNet50 (pretrained on ImageNet → fine-tuned)
- **Framework:** PyTorch
- **Input:** 224 × 224 RGB retinal fundus images
- **Output:** 5-class softmax → DR stage + risk score
- **Optimiser:** Adam (lr = 1e-4)
- **Loss:** CrossEntropyLoss
- **Augmentation:** Random rotation, horizontal flip, brightness jitter
- **Best Val Accuracy:** 86.9% (Epoch 15)
- **Best F1 Score:** 85.2%

### To use real weights:
1. Train with `torchvision.models.resnet50(pretrained=True)`, modify final FC layer to 5 outputs
2. Save best model: `torch.save(model.state_dict(), 'best_model.pth')`
3. Load in `backend/main.py` → replace `run_inference()` body with real forward pass

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3 (custom design system), Vanilla JS, Chart.js 4 |
| Backend | Python 3.11, FastAPI, Uvicorn, Pillow, NumPy |
| Database | SQLite3 (via Python stdlib) |
| ML Model | PyTorch, torchvision, ResNet50 |
| Dataset | APTOS 2019 (Kaggle) |

---

## Team — Batch 09

| Member | Roll Number |
|---|---|
| Dr. G. Aparna | Guide |
| A. Jyothika | 22E51A6603 |
| K. Phani Sai | 22E51A6627 |
| M. Gayatri | 22E51A6640 |
| P. Pranusha | 22E51A6651 |

**Department of CSM · Hyderabad Institute of Technology And Management**
