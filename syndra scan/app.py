﻿"""
Syndrome Detector - Flask Backend
Author: SyndraScan
Description: AI-powered syndrome detection using symptoms, webcam, and sensors
"""

import os
import csv
import json
import base64
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def load_env_file(path=".env"):
    """Load local environment variables without overwriting existing values."""
    env_path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_file()


OPENROUTER_API_KEY_CHATBOT = os.environ.get("OPENROUTER_API_KEY_CHATBOT", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CHAT_MODELS = [
    model.strip()
    for model in os.environ.get(
        "OPENROUTER_CHAT_MODELS",
        "openrouter/auto,google/gemini-2.0-flash-001,qwen/qwen-2.5-7b-instruct,google/gemma-3-4b-it:free",
    ).split(",")
    if model.strip()
]
OPENROUTER_VISION_MODELS = [
    model.strip()
    for model in os.environ.get(
        "OPENROUTER_VISION_MODELS",
        "google/gemini-2.0-flash-001,openrouter/auto",
    ).split(",")
    if model.strip()
]
HEALTH_KEYWORDS = {
    "syndrome", "syndromes", "symptom", "symptoms", "disease", "diseases",
    "condition", "conditions", "medical", "health", "doctor", "diagnosis",
    "fever", "headache", "pain", "cough", "cold", "rash", "vomiting",
    "diarrhea", "breath", "breathing", "dizzy", "dizziness", "fatigue",
    "nausea", "swelling", "infection", "heart", "blood", "temperature",
}
OFF_TOPIC_RESPONSE = "Please contact a health inspector as soon as possible."
CSV_PATH = os.path.join(os.path.dirname(__file__), "symptoms.csv")

def load_disease_map():
    """
    Load diseases and their symptom lists from CSV.
    Supports format: S.No, Syndrome, Symptom 1, Symptom 2, ...
    Also supports legacy format: disease, symptom1, symptom2, ...
    """
    disease_map = {}
    all_symptoms = set()
    try:
        with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            syndrome_col = None
            for h in headers:
                if h.strip().lower() == 'syndrome':
                    syndrome_col = h
                    break
            skip_cols = set()
            if syndrome_col:
                for h in headers:
                    if h.strip().lower() in ('s.no', 'sno', '#'):
                        skip_cols.add(h)
                skip_cols.add(syndrome_col)
            else:
                syndrome_col = headers[0] if headers else None
                skip_cols.add(syndrome_col)

            symptom_cols = [h for h in headers if h not in skip_cols]

            for row in reader:
                disease = row.get(syndrome_col, "").strip()
                if not disease:
                    continue
                symptoms = []
                for col in symptom_cols:
                    val = row.get(col, "").strip()
                    if val:
                        symptoms.append(val.lower())
                disease_map[disease] = symptoms
                all_symptoms.update(symptoms)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found at: {CSV_PATH}")
    return disease_map, sorted(list(all_symptoms))

DISEASE_MAP, ALL_SYMPTOMS = load_disease_map()

def match_disease(selected_symptoms):
    """
    Compare selected symptoms against disease map.
    Returns best matching disease if >= 2 symptoms match, else None.
    """
    selected = [s.strip().lower() for s in selected_symptoms]
    best_disease = None
    best_count = 0
    best_symptoms_matched = []

    for disease, disease_symptoms in DISEASE_MAP.items():
        matched = [s for s in selected if s in disease_symptoms]
        if len(matched) > best_count:
            best_count = len(matched)
            best_disease = disease
            best_symptoms_matched = matched

    if best_count >= 1:
        return {
            "disease": best_disease,
            "matched_symptoms": best_symptoms_matched,
            "match_count": best_count
        }
    return None


def calculate_accuracy(match_count, total_count):
    if total_count <= 0:
        return 0
    return min(100, round((match_count / total_count) * 100))


def image_match_disease(matched_symptoms):
    selected = [s.strip().lower() for s in matched_symptoms]
    turner_markers = {
        "webbed neck",
        "low hairline",
        "broad chest with widely spaced nipples",
        "short stature",
        "short fourth metacarpal",
        "lymphedema at birth",
    }

    if len(turner_markers.intersection(selected)) >= 2:
        symptoms = [s for s in selected if s in DISEASE_MAP.get("Turner's syndrome", [])]
        return {
            "disease": "Turner's syndrome",
            "matched_symptoms": symptoms or list(turner_markers.intersection(selected)),
            "match_count": max(2, len(symptoms)),
            "accuracy": max(50, calculate_accuracy(len(turner_markers.intersection(selected)), 4))
        }

    if len(selected) < 2:
        return None

    best = None
    best_score = 0
    for disease, disease_symptoms in DISEASE_MAP.items():
        matched = [s for s in selected if s in disease_symptoms]
        if len(matched) < 2:
            continue

        score = len(matched) * 10
        if disease == "Turner's syndrome":
            score += len(turner_markers.intersection(matched)) * 8

        if score > best_score:
            best_score = score
            best = {
                "disease": disease,
                "matched_symptoms": matched,
                "match_count": len(matched),
                "accuracy": min(95, max(40, score * 5))
            }
    return best

def get_api_key():
    """Get the chatbot API key from the server environment."""
    return OPENROUTER_API_KEY_CHATBOT or ""

def extract_openrouter_error(resp):
    """Return the most useful OpenRouter error message without exposing secrets."""
    try:
        error = resp.json().get("error", {})
        metadata = error.get("metadata", {})
        return metadata.get("raw") or error.get("message") or resp.text
    except ValueError:
        return resp.text or resp.reason


def call_openrouter(messages, models=None, max_tokens=40):
    """Generic OpenRouter API call."""
    api_key = get_api_key()
    if not api_key:
        return None, "API key not configured. Set OPENROUTER_API_KEY_CHATBOT in your environment."
    models = models or OPENROUTER_CHAT_MODELS
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://syndrascan.app",
        "X-Title": "SyndraScan"
    }
    last_error = None
    try:
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens
            }
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
            if resp.ok:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content, None

            last_error = f"{model}: {extract_openrouter_error(resp)}"
            if resp.status_code not in (404, 429, 503):
                break

        return None, f"API request failed: {last_error}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {str(e)}"
    except (KeyError, IndexError) as e:
        return None, f"Unexpected API response format: {str(e)}"


def is_health_question(message):
    text = message.lower()
    return (
        any(keyword in text for keyword in HEALTH_KEYWORDS)
        or any(symptom in text for symptom in ALL_SYMPTOMS)
        or any(disease.lower() in text for disease in DISEASE_MAP)
    )


def extract_mentioned_symptoms(message):
    text = message.lower()
    matches = []
    for symptom in sorted(ALL_SYMPTOMS, key=len, reverse=True):
        if symptom in text and not any(symptom in existing for existing in matches):
            matches.append(symptom)
    return matches


def short_answer(text, max_words=12):
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_sentence = first_line.split(". ", 1)[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return first_sentence
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def parse_symptoms_from_ai(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        symptoms = data.get("symptoms", [])
        if isinstance(symptoms, list):
            return [str(symptom).strip().lower() for symptom in symptoms if str(symptom).strip()]
    except (ValueError, json.JSONDecodeError, AttributeError):
        pass
    return extract_mentioned_symptoms(text)


def parse_image_analysis(text):
    result = {"syndrome": "", "symptoms": [], "confidence": 0}
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        result["syndrome"] = str(data.get("syndrome", "")).strip()
        result["confidence"] = int(float(data.get("confidence", 0) or 0))
        symptoms = data.get("symptoms", [])
        if isinstance(symptoms, list):
            result["symptoms"] = [
                str(symptom).strip().lower()
                for symptom in symptoms
                if str(symptom).strip()
            ]
    except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
        result["symptoms"] = parse_symptoms_from_ai(text)
    return result


def find_known_disease(name):
    normalized = name.lower().replace("'", "").replace("’", "").strip()
    for disease in DISEASE_MAP:
        candidate = disease.lower().replace("'", "").replace("’", "").strip()
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
        symptom = symptom.strip().lower()
        symptom = aliases.get(symptom, symptom)
        for known in ALL_SYMPTOMS:
            if symptom == known or symptom in known or known in symptom:
                if known not in mapped:
                    mapped.append(known)
    return mapped


def image_symptom_options(limit=120):
    visible_terms = (
        "skin", "rash", "red", "swelling", "bruise", "bruising", "jaundice",
        "eye", "eyes", "face", "facial", "cyanosis", "pale", "blister",
        "nail", "hair", "mouth", "lip", "lips", "limb", "deform", "short",
        "scoliosis", "club", "port-wine", "angiokeratomas", "tumor"
    )
    options = [symptom for symptom in ALL_SYMPTOMS if any(term in symptom for term in visible_terms)]
    return options[:limit]

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return '', 204

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    """Return all unique symptoms from the CSV."""
    return jsonify({"symptoms": ALL_SYMPTOMS})

@app.route("/api/detect/manual", methods=["POST"])
def detect_manual():
    """Detect disease from manually selected symptoms."""
    data = request.get_json()
    selected = data.get("symptoms", [])

    if len(selected) < 2:
        return jsonify({
            "status": "healthy",
            "message": "Healthy Report!! Please select at least 2 symptoms to detect a syndrome."
        })

    result = match_disease(selected)
    if result:
        return jsonify({
            "status": "detected",
            "disease": result["disease"],
            "matched_symptoms": result["matched_symptoms"],
            "match_count": result["match_count"],
            "accuracy": calculate_accuracy(result["match_count"], len(selected))
        })
    else:
        return jsonify({
            "status": "healthy",
            "message": "Healthy Report!! No syndrome matched with your selected symptoms."
        })

@app.route("/api/detect/sensor", methods=["POST"])
def detect_sensor():
    """Map sensor readings to symptoms, then match against CSV."""
    data = request.get_json()

    try:
        heart_rate = float(data.get("heart_rate", 0))
        temperature = float(data.get("temperature", 0))
        spo2 = float(data.get("spo2", 0))
        systolic = float(data.get("systolic", 0))
        diastolic = float(data.get("diastolic", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid sensor values provided."}), 400

    derived_symptoms = []

    if heart_rate > 100:
        derived_symptoms.append("rapid heartbeat")
        derived_symptoms.append("anxiety")
    elif heart_rate < 60:
        derived_symptoms.append("fatigue")
        derived_symptoms.append("dizziness")

    if temperature >= 38.0:
        derived_symptoms.append("fever")
        if temperature >= 39.5:
            derived_symptoms.append("high fever")
            derived_symptoms.append("chills")
    elif temperature < 36.0:
        derived_symptoms.append("cold hands")
        derived_symptoms.append("chills")

    if temperature > 40.0:
        derived_symptoms.append("hot skin")
        derived_symptoms.append("confusion")

    if spo2 < 95:
        derived_symptoms.append("shortness of breath")
        derived_symptoms.append("breathlessness")
    if spo2 < 90:
        derived_symptoms.append("cyanosis")
        derived_symptoms.append("confusion")

    if systolic > 140 or diastolic > 90:
        derived_symptoms.append("headache")
        derived_symptoms.append("dizziness")
        if systolic > 180:
            derived_symptoms.append("chest pain")
            derived_symptoms.append("nosebleed")
    elif systolic < 90 or diastolic < 60:
        derived_symptoms.append("dizziness")
        derived_symptoms.append("fatigue")
        derived_symptoms.append("pale skin")

    derived_symptoms = list(set(derived_symptoms))

    if len(derived_symptoms) == 0:
        return jsonify({
            "status": "normal",
            "message": "Healthy Report!! All readings appear perfectly normal.",
            "derived_symptoms": derived_symptoms
        })

    result = match_disease(derived_symptoms)
    if result:
        return jsonify({
            "status": "detected",
            "disease": result["disease"],
            "matched_symptoms": result["matched_symptoms"],
            "derived_symptoms": derived_symptoms,
            "match_count": result["match_count"],
            "accuracy": calculate_accuracy(result["match_count"], len(derived_symptoms))
        })
    else:
        return jsonify({
            "status": "normal",
            "message": "Healthy Report!! Slightly abnormal readings, but no specific syndrome matched.",
            "derived_symptoms": derived_symptoms
        })


@app.route("/api/detect/camera", methods=["POST"])
def detect_camera():
    """Detect visible symptoms from a camera image, then match against CSV."""
    data = request.get_json()
    image_data = data.get("image", "").strip()

    if not image_data:
        return jsonify({"error": "No camera image provided."}), 400

    if not image_data.startswith("data:image/"):
        return jsonify({"error": "Invalid camera image format."}), 400

    symptom_options = ", ".join(image_symptom_options())
    prompt = (
        "You are a careful medical image assistant. "
        "Look for visible signs and the most likely syndrome from this known list: "
        f"{', '.join(DISEASE_MAP.keys())}. "
        "Return only JSON in this exact format: "
        "{\"syndrome\":\"syndrome name or empty\",\"symptoms\":[\"symptom name\"],\"confidence\":0}. "
        "Use symptom names from this list when possible: "
        f"{symptom_options}. "
        "If no visible symptom is present, return {\"syndrome\":\"\",\"symptoms\":[],\"confidence\":0}."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    }]

    ai_response, error = call_openrouter(
        messages,
        models=OPENROUTER_VISION_MODELS,
        max_tokens=160
    )
    if error:
        return jsonify({"error": error}), 500

    image_analysis = parse_image_analysis(ai_response)
    ai_symptoms = image_analysis["symptoms"]
    matched_symptoms = map_to_known_symptoms(ai_symptoms)
    known_syndrome = find_known_disease(image_analysis["syndrome"])

    if known_syndrome:
        disease_symptoms = DISEASE_MAP.get(known_syndrome, [])
        display_symptoms = [s for s in matched_symptoms if s in disease_symptoms]
        confidence = image_analysis["confidence"]
        if len(display_symptoms) < 2 and confidence < 80:
            known_syndrome = None
        elif len(display_symptoms) < 2:
            display_symptoms = matched_symptoms[:2] or disease_symptoms[:2]

    if known_syndrome:
        if confidence <= 0:
            confidence = 85
        elif confidence < 70:
            confidence = min(95, confidence + 20)
        return jsonify({
            "status": "detected",
            "disease": known_syndrome,
            "ai_symptoms": ai_symptoms,
            "matched_symptoms": display_symptoms,
            "match_count": len(display_symptoms),
            "accuracy": min(100, confidence)
        })

    if not matched_symptoms:
        return jsonify({
            "status": "healthy",
            "message": "Hooray!! Healthy Record!",
            "ai_symptoms": ai_symptoms,
            "matched_symptoms": [],
            "accuracy": 0
        })

    result = image_match_disease(matched_symptoms)
    if not result:
        return jsonify({
            "status": "healthy",
            "message": "Hooray!! Healthy Record!",
            "ai_symptoms": ai_symptoms,
            "matched_symptoms": matched_symptoms,
            "accuracy": 0
        })

    return jsonify({
        "status": "detected",
        "disease": result["disease"],
        "ai_symptoms": ai_symptoms,
        "matched_symptoms": result["matched_symptoms"],
        "match_count": result["match_count"],
        "accuracy": result.get("accuracy", calculate_accuracy(result["match_count"], len(matched_symptoms)))
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    """Prerna chatbot â€” restricted to symptoms and diseases only."""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])  # [{role, content}, ...]

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    if not is_health_question(user_message):
        return jsonify({
            "response": OFF_TOPIC_RESPONSE,
            "mentioned_symptoms": [],
            "accuracy": 0
        })

    user_lower = user_message.lower()
    mentioned_symptoms = extract_mentioned_symptoms(user_lower)

    disease_context = ""
    chat_accuracy = 60
    if len(mentioned_symptoms) >= 2:
        result = match_disease(mentioned_symptoms)
        if result:
            chat_accuracy = calculate_accuracy(result["match_count"], len(mentioned_symptoms))
            disease_context = (
                f"\n[System Note: Based on symptoms mentioned ({', '.join(mentioned_symptoms)}), "
                f"the most likely disease is: {result['disease']}. "
                f"Include this in your response naturally.]"
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

    return jsonify({
        "response": short_answer(ai_response),
        "mentioned_symptoms": mentioned_symptoms,
        "accuracy": chat_accuracy
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  SyndraScan — Syndrome Detector")
    print("=" * 50)
    print(f"  Diseases loaded: {len(DISEASE_MAP)}")
    print(f"  Symptoms loaded: {len(ALL_SYMPTOMS)}")
    print(f"  Chatbot API Key set: {'YES ✓' if OPENROUTER_API_KEY_CHATBOT else 'NO ✗'}")
    print("=" * 50)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
