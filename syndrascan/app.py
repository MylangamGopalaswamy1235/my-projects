import os
import csv
import json
import requests
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

def load_env_file(path=".env"):
    env_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file()

OPENROUTER_API_KEY_CHATBOT = os.environ.get("OPENROUTER_API_KEY_CHATBOT", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CHAT_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_CHAT_MODELS",
        "openrouter/auto,google/gemini-2.0-flash-001,qwen/qwen-2.5-7b-instruct,google/gemma-3-4b-it:free",
    ).split(",") if m.strip()
]
OPENROUTER_VISION_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_VISION_MODELS",
        "google/gemini-2.0-flash-001,openrouter/auto",
    ).split(",") if m.strip()
]

HEALTH_KEYWORDS = {
    "syndrome","syndromes","symptom","symptoms","disease","diseases",
    "condition","conditions","medical","health","doctor","diagnosis",
    "fever","headache","pain","cough","cold","rash","vomiting",
    "diarrhea","breath","breathing","dizzy","dizziness","fatigue",
    "nausea","swelling","infection","heart","blood","temperature",
}
OFF_TOPIC_RESPONSE = "Please contact a health inspector as soon as possible."
CSV_PATH = os.path.join(BASE_DIR, "symptoms.csv")

def load_disease_map():
    disease_map = {}
    all_symptoms = set()
    try:
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            syndrome_col = next((h for h in headers if h.strip().lower() == "syndrome"), None)
            skip_cols = set()
            if syndrome_col:
                for h in headers:
                    if h.strip().lower() in ("s.no", "sno", "#"):
                        skip_cols.add(h)
                skip_cols.add(syndrome_col)
            else:
                syndrome_col = headers[0] if headers else None
                if syndrome_col:
                    skip_cols.add(syndrome_col)
            symptom_cols = [h for h in headers if h not in skip_cols]
            for row in reader:
                disease = row.get(syndrome_col, "").strip()
                if not disease:
                    continue
                symptoms = [row[col].strip().lower() for col in symptom_cols if row.get(col, "").strip()]
                disease_map[disease] = symptoms
                all_symptoms.update(symptoms)
    except FileNotFoundError:
        print(f"[ERROR] symptoms.csv not found at: {CSV_PATH}")
    return disease_map, sorted(all_symptoms)

DISEASE_MAP, ALL_SYMPTOMS = load_disease_map()

def match_disease(selected_symptoms):
    selected = [s.strip().lower() for s in selected_symptoms]
    best_disease, best_count, best_matched = None, 0, []
    for disease, dsymptoms in DISEASE_MAP.items():
        matched = [s for s in selected if s in dsymptoms]
        if len(matched) > best_count:
            best_count = len(matched)
            best_disease = disease
            best_matched = matched
    if best_count >= 1:
        return {"disease": best_disease, "matched_symptoms": best_matched, "match_count": best_count}
    return None

def calculate_accuracy(match_count, total_count):
    if total_count <= 0:
        return 0
    return min(100, round((match_count / total_count) * 100))

def image_match_disease(matched_symptoms):
    selected = [s.strip().lower() for s in matched_symptoms]
    turner_markers = {
        "webbed neck","low hairline","broad chest with widely spaced nipples",
        "short stature","short fourth metacarpal","lymphedema at birth",
    }
    if len(turner_markers & set(selected)) >= 2:
        symptoms = [s for s in selected if s in DISEASE_MAP.get("Turner's syndrome", [])]
        return {
            "disease": "Turner's syndrome",
            "matched_symptoms": symptoms or list(turner_markers & set(selected)),
            "match_count": max(2, len(symptoms)),
            "accuracy": max(50, calculate_accuracy(len(turner_markers & set(selected)), 4)),
        }
    if len(selected) < 2:
        return None
    best, best_score = None, 0
    for disease, dsymptoms in DISEASE_MAP.items():
        matched = [s for s in selected if s in dsymptoms]
        if len(matched) < 2:
            continue
        score = len(matched) * 10
        if disease == "Turner's syndrome":
            score += len(turner_markers & set(matched)) * 8
        if score > best_score:
            best_score = score
            best = {
                "disease": disease,
                "matched_symptoms": matched,
                "match_count": len(matched),
                "accuracy": min(95, max(40, score * 5)),
            }
    return best

def extract_openrouter_error(resp):
    try:
        err = resp.json().get("error", {})
        return err.get("metadata", {}).get("raw") or err.get("message") or resp.text
    except ValueError:
        return resp.text or resp.reason

def call_openrouter(messages, models=None, max_tokens=40):
    api_key = OPENROUTER_API_KEY_CHATBOT
    if not api_key:
        return None, "API key not configured. Set OPENROUTER_API_KEY_CHATBOT in environment."
    models = models or OPENROUTER_CHAT_MODELS
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://syndrascan.vercel.app",
        "X-Title": "SyndraScan",
    }
    last_error = None
    try:
        for model in models:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=30,
            )
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"], None
            last_error = f"{model}: {extract_openrouter_error(resp)}"
            if resp.status_code not in (404, 429, 503):
                break
        return None, f"API request failed: {last_error}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except (KeyError, IndexError) as e:
        return None, f"Unexpected API response: {str(e)}"

def is_health_question(message):
    text = message.lower()
    return (
        any(kw in text for kw in HEALTH_KEYWORDS)
        or any(s in text for s in ALL_SYMPTOMS)
        or any(d.lower() in text for d in DISEASE_MAP)
    )

def extract_mentioned_symptoms(message):
    text = message.lower()
    matches = []
    for symptom in sorted(ALL_SYMPTOMS, key=len, reverse=True):
        if symptom in text and not any(symptom in existing for existing in matches):
            matches.append(symptom)
    return matches

def short_answer(text, max_words=12):
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    first_sentence = first_line.split(". ", 1)[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return first_sentence
    return " ".join(words[:max_words]).rstrip(".,;:") + "."

def parse_image_analysis(text):
    result = {"syndrome": "", "symptoms": [], "confidence": 0}
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
        result["syndrome"] = str(data.get("syndrome", "")).strip()
        result["confidence"] = int(float(data.get("confidence", 0) or 0))
        syms = data.get("symptoms", [])
        if isinstance(syms, list):
            result["symptoms"] = [str(s).strip().lower() for s in syms if str(s).strip()]
    except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
        result["symptoms"] = extract_mentioned_symptoms(text)
    return result

def find_known_disease(name):
    normalized = name.lower().replace("'", "").replace("\u2019", "").strip()
    for disease in DISEASE_MAP:
        candidate = disease.lower().replace("'", "").replace("\u2019", "").strip()
        if normalized == candidate or normalized in candidate or candidate in normalized:
            return disease
    return None

def map_to_known_symptoms(symptoms):
    aliases = {
        "wide neck": "webbed neck",
        "neck webbing": "webbed neck",
        "low posterior hairline": "low hairline",
        "broad chest": "broad chest with widely spaced nipples",
        "widely spaced nipples": "broad chest with widely spaced nipples",
        "short height": "short stature",
        "small stature": "short stature",
        "puffy hands": "lymphedema at birth",
        "puffy feet": "lymphedema at birth",
    }
    mapped = []
    for symptom in symptoms:
        symptom = aliases.get(symptom.strip().lower(), symptom.strip().lower())
        for known in ALL_SYMPTOMS:
            if (symptom == known or symptom in known or known in symptom) and known not in mapped:
                mapped.append(known)
    return mapped

def image_symptom_options(limit=120):
    visible_terms = (
        "skin","rash","red","swelling","bruise","bruising","jaundice",
        "eye","eyes","face","facial","cyanosis","pale","blister",
        "nail","hair","mouth","lip","lips","limb","deform","short",
        "scoliosis","club","port-wine","angiokeratomas","tumor",
    )
    return [s for s in ALL_SYMPTOMS if any(t in s for t in visible_terms)][:limit]

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": ALL_SYMPTOMS})

@app.route("/api/detect/manual", methods=["POST"])
def detect_manual():
    data = request.get_json(force=True)
    selected = data.get("symptoms", [])
    if len(selected) < 2:
        return jsonify({"status": "healthy", "message": "Healthy Report!! Please select at least 2 symptoms to detect a syndrome."})
    result = match_disease(selected)
    if result:
        return jsonify({
            "status": "detected",
            "disease": result["disease"],
            "matched_symptoms": result["matched_symptoms"],
            "match_count": result["match_count"],
            "accuracy": calculate_accuracy(result["match_count"], len(selected)),
        })
    return jsonify({"status": "healthy", "message": "Healthy Report!! No syndrome matched with your selected symptoms."})

@app.route("/api/detect/sensor", methods=["POST"])
def detect_sensor():
    data = request.get_json(force=True)
    try:
        heart_rate = float(data.get("heart_rate") or 0)
        temperature = float(data.get("temperature") or 0)
        spo2 = float(data.get("spo2") or 0)
        systolic = float(data.get("systolic") or 0)
        diastolic = float(data.get("diastolic") or 0)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid sensor values provided."}), 400

    derived = []
    if heart_rate > 100:
        derived += ["rapid heartbeat", "anxiety"]
    elif 0 < heart_rate < 60:
        derived += ["fatigue", "dizziness"]
    if temperature >= 38.0:
        derived.append("fever")
        if temperature >= 39.5:
            derived += ["high fever", "chills"]
    elif 0 < temperature < 36.0:
        derived += ["cold hands", "chills"]
    if temperature > 40.0:
        derived += ["hot skin", "confusion"]
    if spo2 and spo2 < 95:
        derived += ["shortness of breath", "breathlessness"]
    if spo2 and spo2 < 90:
        derived += ["cyanosis", "confusion"]
    if systolic > 140 or diastolic > 90:
        derived += ["headache", "dizziness"]
        if systolic > 180:
            derived += ["chest pain", "nosebleed"]
    elif 0 < systolic < 90 or (0 < diastolic < 60):
        derived += ["dizziness", "fatigue", "pale skin"]

    derived = list(set(derived))
    if not derived:
        return jsonify({"status": "normal", "message": "Healthy Report!! All readings appear perfectly normal.", "derived_symptoms": []})

    result = match_disease(derived)
    if result:
        return jsonify({
            "status": "detected",
            "disease": result["disease"],
            "matched_symptoms": result["matched_symptoms"],
            "derived_symptoms": derived,
            "match_count": result["match_count"],
            "accuracy": calculate_accuracy(result["match_count"], len(derived)),
        })
    return jsonify({"status": "normal", "message": "Healthy Report!! Slightly abnormal readings, but no specific syndrome matched.", "derived_symptoms": derived})

@app.route("/api/detect/camera", methods=["POST"])
def detect_camera():
    data = request.get_json(force=True)
    image_data = (data.get("image") or "").strip()
    if not image_data:
        return jsonify({"error": "No camera image provided."}), 400
    if not image_data.startswith("data:image/"):
        return jsonify({"error": "Invalid camera image format."}), 400

    symptom_options = ", ".join(image_symptom_options())
    prompt = (
        "You are a careful medical image assistant. "
        "Look for visible signs and the most likely syndrome from this known list: "
        f"{', '.join(DISEASE_MAP.keys())}. "
        'Return ONLY JSON: {"syndrome":"name or empty","symptoms":["symptom"],"confidence":0}. '
        f"Use symptom names from: {symptom_options}. "
        'If nothing visible, return {"syndrome":"","symptoms":[],"confidence":0}.'
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data}},
    ]}]
    ai_response, error = call_openrouter(messages, models=OPENROUTER_VISION_MODELS, max_tokens=160)
    if error:
        return jsonify({"error": error}), 500

    analysis = parse_image_analysis(ai_response)
    matched = map_to_known_symptoms(analysis["symptoms"])
    known = find_known_disease(analysis["syndrome"])

    if known:
        dsyms = DISEASE_MAP.get(known, [])
        display = [s for s in matched if s in dsyms]
        confidence = analysis["confidence"] or 85
        if len(display) < 2 and confidence < 80:
            known = None
        else:
            if len(display) < 2:
                display = matched[:2] or dsyms[:2]
            return jsonify({
                "status": "detected",
                "disease": known,
                "ai_symptoms": analysis["symptoms"],
                "matched_symptoms": display,
                "match_count": len(display),
                "accuracy": min(100, confidence),
            })

    if not matched:
        return jsonify({"status": "healthy", "message": "Hooray!! Healthy Record!", "ai_symptoms": analysis["symptoms"], "matched_symptoms": [], "accuracy": 0})

    result = image_match_disease(matched)
    if not result:
        return jsonify({"status": "healthy", "message": "Hooray!! Healthy Record!", "ai_symptoms": analysis["symptoms"], "matched_symptoms": matched, "accuracy": 0})

    return jsonify({
        "status": "detected",
        "disease": result["disease"],
        "ai_symptoms": analysis["symptoms"],
        "matched_symptoms": result["matched_symptoms"],
        "match_count": result["match_count"],
        "accuracy": result.get("accuracy", calculate_accuracy(result["match_count"], len(matched))),
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error": "Empty message."}), 400
    if not is_health_question(user_message):
        return jsonify({"response": OFF_TOPIC_RESPONSE, "mentioned_symptoms": [], "accuracy": 0})

    mentioned = extract_mentioned_symptoms(user_message.lower())
    disease_context = ""
    chat_accuracy = 60
    if len(mentioned) >= 2:
        result = match_disease(mentioned)
        if result:
            chat_accuracy = calculate_accuracy(result["match_count"], len(mentioned))
            disease_context = (
                f"\n[System Note: Based on symptoms mentioned ({', '.join(mentioned)}), "
                f"the most likely disease is: {result['disease']}. Include this in your response naturally.]"
            )

    system_prompt = (
        "You are Prerna, a medical assistant chatbot for SyndraScan. "
        "Answer only about syndromes, symptoms, and diseases. "
        "Keep every answer under 12 words. "
        "Answer only the user's current message. "
        "Do not write examples, paragraphs, lists, or long explanations. "
        "For any unrelated question, reply exactly: Please contact a health inspector as soon as possible. "
        "When symptoms are described, name the likely disease or ask for more symptoms. "
        "For diagnosis questions, say: Consult a doctor."
        + disease_context
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    ai_response, error = call_openrouter(messages)
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"response": short_answer(ai_response), "mentioned_symptoms": mentioned, "accuracy": chat_accuracy})

if __name__ == "__main__":
    print("=" * 50)
    print("  SyndraScan — Syndrome Detector")
    print(f"  Diseases: {len(DISEASE_MAP)}  |  Symptoms: {len(ALL_SYMPTOMS)}")
    print(f"  API Key: {'SET' if OPENROUTER_API_KEY_CHATBOT else 'NOT SET'}")
    print("=" * 50)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
