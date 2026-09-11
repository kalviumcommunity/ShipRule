# 🚀 Deployment Guide — ShipRule CDLP

This guide provides step-by-step instructions for deploying the **ShipRule** platform:
- **Backend (FastAPI)**: Deployed on **Render** as a Web Service.
- **Frontend (Next.js)**: Deployed on **Vercel**.

---

## 🐍 1. Backend Deployment — Render

### Overview
The backend is built with FastAPI and Uvicorn. Both root `/requirements.txt` and `/server/requirements.txt` are configured with all required dependencies including `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, and `pymongo`.

### Step-by-Step Render Setup

1. **Log in to Render**
   - Go to [https://dashboard.render.com](https://dashboard.render.com) and click **New +** -> **Web Service**.

2. **Connect Repository**
   - Select your GitHub repository (`ShipRule`).

3. **Configure Service Settings**
   - **Name**: `shiprule-backend` (or your choice)
   - **Region**: Select closest region (e.g., Singapore or Oregon)
   - **Branch**: `main` (or active branch)
   - **Root Directory**: `server` (or leave empty if using workspace root)
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
     *(Note: If Root Directory is set to `server`, `main:app` will correctly find `server/main.py`)*

4. **Environment Variables on Render**
   Under **Environment Variables**, add the following:
   - `GROQ_API_KEY`: *(Your Groq API key for LLM inference)*
   - `MONGODB_URI`: *(Your MongoDB connection string, e.g. `mongodb+srv://...`)*
   - `JWT_SECRET`: *(Secret key for JWT token generation)*
   - `CORS_ORIGINS`: `https://your-shiprule-frontend.vercel.app` *(or `*` during initial setup)*
   - `PYTHON_VERSION`: `3.11.8` *(Recommended for optimal package compatibility)*

5. **Resolving `ModuleNotFoundError: No module named 'fastapi'`**
   - If Render encounters this error during startup, ensure `requirements.txt` is committed at both the repository root `/` and inside `/server/`.
   - Ensure the **Build Command** is set to `pip install -r requirements.txt` so pip installs all dependencies into the virtual environment before Uvicorn starts.

---

## ⚡ 2. Frontend Deployment — Vercel

### Overview
The frontend is a Next.js App Router application located inside the `/client` directory.

### Step-by-Step Vercel Setup

1. **Log in to Vercel**
   - Go to [https://vercel.com/dashboard](https://vercel.com/dashboard) and click **Add New...** -> **Project**.

2. **Import Repository**
   - Select your GitHub repository (`ShipRule`).

3. **Configure Project Settings**
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click **Edit** and set to `client`
   - **Build Command**: `npm run build` *(Default)*
   - **Output Directory**: `.next` *(Default)*
   - **Install Command**: `npm install` *(Default)*

4. **Environment Variables on Vercel**
   - `NEXT_PUBLIC_API_URL`: `https://shiprule-backend.onrender.com` *(Replace with your deployed Render backend URL)*

5. **Deploy**
   - Click **Deploy**. Vercel will build and publish your Next.js application.

---

## 🔍 3. Verification & Health Check

1. **Backend Verification**:
   - Open `https://shiprule-backend.onrender.com/health`
   - Response should be `{"status": "healthy", ...}`
   - Open Swagger docs: `https://shiprule-backend.onrender.com/docs`

2. **Frontend Verification**:
   - Open your Vercel URL (e.g. `https://shiprule.vercel.app`)
   - Test login, query execution, splash page transition, profile dropdown, and settings.