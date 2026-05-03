# SyndraScan — Syndrome Detector

A full-stack medical symptom detection web app using Flask + AI.

---

## Project Structure

```
syndrascan/
├── app.py                 # Flask backend (Vercel entry point)
├── symptoms.csv           # Disease-symptom database (100 syndromes)
├── requirements.txt       # Python dependencies
├── vercel.json            # Vercel deployment config
├── .env.example           # Environment variable template
├── .gitignore
├── .vercelignore
├── templates/
│   └── index.html         # Main frontend
└── static/
    ├── style.css
    └── app.js
```

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Edit .env and add your real API key

# 3. Run locally
python app.py
# Visit http://localhost:5000
```

---

## Deploy to Vercel

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USER/syndrascan.git
git push -u origin main
```

### 2. Import on Vercel
- Go to https://vercel.com/new
- Import your GitHub repo
- **IMPORTANT:** Do NOT add .env file — set env vars in dashboard instead

### 3. Set Environment Variables in Vercel Dashboard
- Go to your project → Settings → Environment Variables
- Add: `OPENROUTER_API_KEY_CHATBOT` = your OpenRouter API key
- This is required — the chatbot and camera features won't work without it

### 4. Deploy
Vercel auto-deploys on every push to `main`.

---

## Features

| Feature | Description |
|---------|-------------|
| Manual Detector | Select symptoms from the CSV list |
| AI Camera | Webcam image analyzed by multimodal AI |
| Sensor Input | Heart rate, temperature, SpO2, blood pressure |
| Prerna Chatbot | Medical AI assistant (restricted to health topics) |

---

## Common Vercel Errors & Fixes

| Error | Fix |
|-------|-----|
| `No module named flask` | Check `requirements.txt` is at repo root |
| `404 on /api/*` | Verify `vercel.json` routes are correct |
| Chatbot returns "API key not configured" | Add env var in Vercel dashboard → Settings → Env Variables |
| `symptoms.csv not found` | Ensure `symptoms.csv` is committed (not in .gitignore) |
| Static files 404 | Ensure `static/` folder is committed with files inside |

---

## ⚠ Disclaimer

Educational tool only. Never substitute for professional medical advice.
