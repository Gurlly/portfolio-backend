# 📡 Job Offer Spam Detector (FastAPI Backend)

This repository contains the **FastAPI backend** for my portfolio project: a **job offer spam detection system**.  
It exposes a REST API that classifies job offers as **legitimate** or **spam**, powered by a trained machine learning model.  
The backend is deployed on **Render** for live demonstration and integrates seamlessly with a Next.js frontend.

---

## 🚀 Features
- **FastAPI** backend with modular structure (`api/`, `core/`, `utils/`)
- **Pre-trained ML model** for spam/ham classification
- **Tokenizer** for text preprocessing
- **REST API endpoints** for predictions
- **Deployed on Render** with production-ready configuration

---

## ⚙️ Setup & Installation
### 1. Clone the repository
```bash
git clone https://github.com/Gurlly/portfolio-backend.git
cd portfolio/backend
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Create a .env file in the `backend/` directory.

`SECRET_KEY=your-secret-key`
`MODEL_PATH=./model/spam-ham-detection-best-...`
`TOKENIZER_PATH=./model/tokenizer.json`

### 5. Run the server locally

```bash
uvicorn app.main:app --reload
```

## 🛠️ Tech Stack
- FastAPI
- PyTorch and scikit-learn
- Uvicorn
- Render 