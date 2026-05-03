# SyndraScan â€” Syndrome Detector

A full-stack medical symptom detection web app using Flask + AI.

---

## ðŸ“ Project Structure

```
syndra scan/
â”œâ”€â”€ app.py                  # Flask backend
â”œâ”€â”€ symptoms.csv            # Disease-symptom database
â”œâ”€â”€ requirements.txt        # Python dependencies
â”œâ”€â”€ README.md
â”œâ”€â”€ templates/
â”‚   â””â”€â”€ index.html          # Main frontend template
â””â”€â”€ static/
    â”œâ”€â”€ css/
    â”‚   â””â”€â”€ style.css       # Styles
    â””â”€â”€ js/
        â””â”€â”€ app.js          # Frontend logic
```

---

## ðŸš€ Setup & Run

### 1. Install Dependencies

```bash
cd "syndra scan"
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Linux/Mac
export OPENROUTER_API_KEY_CHATBOT=your_key_here

# Windows
set OPENROUTER_API_KEY_CHATBOT=your_key_here
```

Get a free API key at: https://openrouter.ai

### 3. Run the App

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## ðŸ”‘ API Key

- Store your OpenRouter API key in `OPENROUTER_API_KEY_CHATBOT`
- Do not commit API keys to the codebase
- Required for: Prerna chatbot

---

## ðŸ§  Features

| Feature | Description |
|---------|-------------|
| **Manual Detector** | Select symptoms from CSV-loaded list |
| **AI Camera** | Webcam feed analyzed by multimodal AI |
| **Sensor Input** | Enter heart rate, temp, SpO2, BP |
| **Prerna Chatbot** | Domain-restricted medical AI assistant |

---

## ðŸ“‹ Disease Detection Logic

- Requires **â‰¥ 2 symptoms** to match any disease
- Uses `symptoms.csv` as the **only source of truth**
- Finds the disease with the **most matching symptoms**

---

## âš ï¸ Disclaimer

This is an **educational tool only**. Never use it as a substitute for professional medical advice.

