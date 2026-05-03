"use strict";

const STATE = {
    selectedSymptoms: new Set(),
    allSymptoms: [],
    chatHistory: [],
    detectionHistory: []
};

const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    symptomGrid: $("symptomGrid"),
    symptomSearch: $("symptomSearch"),
    selectedTags: $("selectedTags"),
    tagCount: $("tagCount"),
    btnManualDetect: $("btnManualDetect"),
    btnClearSymptoms: $("btnClearSymptoms"),
    heartRate: $("heartRate"),
    bodyTemp: $("bodyTemp"),
    spo2: $("spo2"),
    systolic: $("systolic"),
    diastolic: $("diastolic"),
    btnSensorDetect: $("btnSensorDetect"),
    cameraVideo: $("cameraVideo"),
    cameraPreview: $("cameraPreview"),
    cameraCanvas: $("cameraCanvas"),
    cameraPlaceholder: $("cameraPlaceholder"),
    btnCameraStart: $("btnCameraStart"),
    btnCameraStop: $("btnCameraStop"),
    btnCameraAnalyze: $("btnCameraAnalyze"),
    cameraUpload: $("cameraUpload"),
    resultCard: $("resultCard"),
    resultContent: $("resultContent"),
    resultSource: $("resultSource"),
    resultDisease: $("resultDisease"),
    resultAccuracy: $("resultAccuracy"),
    resultSymptomTags: $("resultSymptomTags"),
    resultError: $("resultError"),
    resultErrorMsg: $("resultErrorMsg"),
    historyList: $("historyList"),
    loadingOverlay: $("loadingOverlay"),
    loadingText: $("loadingText"),
    chatbotFab: $("chatbotFab"),
    chatbotPanel: $("chatbotPanel"),
    chatClose: $("chatClose"),
    chatMessages: $("chatMessages"),
    chatInput: $("chatInput"),
    chatSend: $("chatSend"),
    fabBadge: $("fabBadge")
};

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadSymptoms();
    initManualDetector();
    initSensorDetector();
    initCameraDetector();
    initChatbot();
});

function initTabs() {
    $$(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            $$(".tab-btn").forEach(b => b.classList.remove("active"));
            $$(".tab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            $(`tab-${btn.dataset.tab}`).classList.add("active");
        });
    });
}

async function loadSymptoms() {
    try {
        const resp = await fetch("/api/symptoms");
        const data = await resp.json();
        STATE.allSymptoms = data.symptoms || [];
        renderSymptomChips(STATE.allSymptoms);
    } catch (err) {
        DOM.symptomGrid.innerHTML = `<p style="color:var(--red);font-size:0.8rem;font-family:var(--font-mono);">Failed to load symptoms. Is Flask running?</p>`;
    }
}

function renderSymptomChips(symptoms) {
    DOM.symptomGrid.innerHTML = "";
    if (!symptoms.length) {
        DOM.symptomGrid.innerHTML = `<p style="color:var(--text-dim);font-size:0.8rem;">No symptoms found.</p>`;
        return;
    }
    symptoms.forEach(symptom => {
        const chip = document.createElement("label");
        chip.className = "symptom-chip" + (STATE.selectedSymptoms.has(symptom) ? " selected" : "");
        chip.dataset.symptom = symptom;
        
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "symptom-checkbox";
        checkbox.value = symptom;
        checkbox.checked = STATE.selectedSymptoms.has(symptom);
        
        const text = document.createElement("span");
        text.textContent = symptom;
        
        chip.appendChild(checkbox);
        chip.appendChild(text);
        
        checkbox.addEventListener("change", (e) => toggleSymptom(symptom, chip, e.target));
        DOM.symptomGrid.appendChild(chip);
    });
}

function toggleSymptom(symptom, chipEl, checkbox) {
    const isChecked = checkbox ? checkbox.checked : !STATE.selectedSymptoms.has(symptom);
    
    if (isChecked) {
        STATE.selectedSymptoms.add(symptom);
        chipEl.classList.add("selected");
    } else {
        STATE.selectedSymptoms.delete(symptom);
        chipEl.classList.remove("selected");
    }
    if (checkbox) {
        checkbox.checked = isChecked;
    }
    updateTagCount();
}

function updateTagCount() {
    const n = STATE.selectedSymptoms.size;
    DOM.tagCount.textContent = `${n} symptom${n !== 1 ? "s" : ""}`;
    DOM.tagCount.style.background = n >= 2
        ? "rgba(0, 229, 200, 0.15)"
        : "rgba(90, 122, 150, 0.1)";
}

function initManualDetector() {
    // Search filter
    DOM.symptomSearch.addEventListener("input", (e) => {
        const q = e.target.value.toLowerCase();
        const filtered = q
            ? STATE.allSymptoms.filter(s => s.includes(q))
            : STATE.allSymptoms;
        renderSymptomChips(filtered);
    });

    // Detect button
    DOM.btnManualDetect.addEventListener("click", async () => {
        const symptoms = Array.from(STATE.selectedSymptoms);
        if (symptoms.length < 2) {
            showResultError("Healthy Report!! Please select at least 2 symptoms to detect a syndrome.", "healthy");
            return;
        }
        showLoading("Analyzing symptoms...");
        try {
            const resp = await fetch("/api/detect/manual", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symptoms })
            });
            const data = await resp.json();
            hideLoading();
            if (resp.ok && data.status === "detected") {
                showResult(data.disease, data.matched_symptoms, "Manual Input", data.accuracy);
                addHistory(data.disease, "manual");
            } else if (data.status === "healthy") {
                showResultError(data.message, "healthy");
            } else {
                showResultError(data.error || data.message || "No disease detected.");
            }
        } catch (err) {
            hideLoading();
            showResultError("Network error. Please check your connection.");
        }
    });

    // Clear button
    DOM.btnClearSymptoms.addEventListener("click", () => {
        STATE.selectedSymptoms.clear();
        $$(".symptom-chip").forEach(c => {
            c.classList.remove("selected");
            const cb = c.querySelector(".symptom-checkbox");
            if (cb) cb.checked = false;
        });
        updateTagCount();
        resetResult();
    });
}

function initSensorDetector() {
    DOM.btnSensorDetect.addEventListener("click", async () => {
        const hr = DOM.heartRate.value;
        const temp = DOM.bodyTemp.value;
        const spo2 = DOM.spo2.value;
        const sys = DOM.systolic.value;
        const dia = DOM.diastolic.value;

        if (!hr && !temp && !spo2 && !sys && !dia) {
            showResultError("Please enter at least one sensor reading.");
            return;
        }

        showLoading("Processing sensor data...");
        try {
            const resp = await fetch("/api/detect/sensor", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    heart_rate: hr || 0,
                    temperature: temp || 0,
                    spo2: spo2 || 0,
                    systolic: sys || 0,
                    diastolic: dia || 0
                })
            });
            const data = await resp.json();
            hideLoading();

            if (data.status === "detected") {
                showResult(data.disease, data.matched_symptoms, "Sensor Analysis", data.accuracy);
                addHistory(data.disease, "sensor");
            } else if (data.status === "normal") {
                showResultError(data.message, "healthy");
            } else {
                showResultError(data.message || data.error || "No match found.");
            }
        } catch (err) {
            hideLoading();
            showResultError("Network error. Please check your connection.");
        }
    });
}

function initCameraDetector() {
    DOM.btnCameraStart.addEventListener("click", startCamera);
    DOM.btnCameraStop.addEventListener("click", stopCamera);
    DOM.btnCameraAnalyze.addEventListener("click", analyzeCameraImage);
    DOM.cameraUpload.addEventListener("change", handleImageUpload);
}

async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showResultError("Camera access is not supported in this browser.");
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });
        DOM.cameraVideo.srcObject = stream;
        DOM.cameraPreview.style.display = "none";
        DOM.cameraPreview.removeAttribute("src");
        DOM.cameraPlaceholder.style.display = "none";
        DOM.btnCameraStart.textContent = "CAMERA ON";
    } catch (err) {
        showResultError("Camera permission was blocked.");
    }
}

function stopCamera() {
    const stream = DOM.cameraVideo.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        DOM.cameraVideo.srcObject = null;
    }
    DOM.btnCameraStart.textContent = "START";
    if (!DOM.cameraPreview.getAttribute("src")) {
        DOM.cameraPlaceholder.style.display = "flex";
    }
}

function handleImageUpload() {
    const file = DOM.cameraUpload.files && DOM.cameraUpload.files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showResultError("Please upload an image file.");
        DOM.cameraUpload.value = "";
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        stopCamera();
        DOM.cameraPreview.src = reader.result;
        DOM.cameraPreview.style.display = "block";
        DOM.cameraPlaceholder.style.display = "none";
        analyzeCameraImage();
    };
    reader.readAsDataURL(file);
}

async function analyzeCameraImage() {
    const image = getCameraImage();
    if (!image) return;

    showLoading("Analyzing camera image...");
    try {
        const resp = await fetch("/api/detect/camera", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image })
        });
        const data = await resp.json();
        hideLoading();

        if (resp.ok && data.status === "detected") {
            showResult(data.disease, data.matched_symptoms, "AI Camera", data.accuracy);
            addHistory(data.disease, "camera");
        } else if (resp.ok && data.status === "healthy") {
            showResultError(data.message || "Healthy Record!", "healthy");
        } else {
            showResultError(data.error || "Camera analysis failed.");
        }
    } catch (err) {
        hideLoading();
        showResultError("Network error. Please check your connection.");
    }
}

function getCameraImage() {
    const uploadedImage = DOM.cameraPreview.getAttribute("src");
    if (uploadedImage) {
        return uploadedImage;
    }

    if (!DOM.cameraVideo.srcObject || DOM.cameraVideo.readyState < 2) {
        showResultError("Start the camera or upload an image.");
        return "";
    }

    const canvas = DOM.cameraCanvas;
    const video = DOM.cameraVideo;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
}

function showResult(disease, symptoms, source, accuracy) {
    // Hide idle + error states
    document.querySelector(".result-idle").style.display = "none";
    DOM.resultError.style.display = "none";

    // Build symptom tags
    DOM.resultSymptomTags.innerHTML = symptoms
        .map(s => `<span class="result-tag">${s}</span>`)
        .join("");

    DOM.resultDisease.textContent = disease;
    DOM.resultAccuracy.textContent = Number.isFinite(Number(accuracy)) ? `${accuracy}%` : "--";
    DOM.resultSource.textContent = `Source: ${source}`;
    DOM.resultContent.style.display = "flex";

    // Pulse the card border
    DOM.resultCard.style.borderColor = "var(--accent)";
    setTimeout(() => { DOM.resultCard.style.borderColor = ""; }, 1200);

    // On smaller screens, scroll down to the result automatically
    if (window.innerWidth <= 900) { DOM.resultCard.scrollIntoView({ behavior: "smooth" }); }
}

function showResultError(msg, type = "error") {
    document.querySelector(".result-idle").style.display = "none";
    DOM.resultContent.style.display = "none";
    DOM.resultErrorMsg.textContent = msg;
    
    const iconEl = DOM.resultError.querySelector(".error-icon");
    if (type === "healthy") {
        iconEl.textContent = "â™¥";
        iconEl.style.color = "var(--green)";
        DOM.resultErrorMsg.style.color = "var(--green)";
    } else {
        iconEl.textContent = "âš ";
        iconEl.style.color = "var(--yellow)";
        DOM.resultErrorMsg.style.color = "var(--text-dim)";
    }

    DOM.resultError.style.display = "flex";

    // On smaller screens, scroll down to the error automatically
    if (window.innerWidth <= 900) { DOM.resultCard.scrollIntoView({ behavior: "smooth" }); }
}

function resetResult() {
    document.querySelector(".result-idle").style.display = "flex";
    DOM.resultContent.style.display = "none";
    DOM.resultError.style.display = "none";
}

function addHistory(disease, source) {
    const item = { disease, source, time: new Date().toLocaleTimeString() };
    STATE.detectionHistory.unshift(item);

    const list = DOM.historyList;
    // Remove "empty" placeholder
    const empty = list.querySelector(".history-empty");
    if (empty) empty.remove();

    const el = document.createElement("div");
    el.className = "history-item";
    const badgeClass = { manual: "badge-manual", sensor: "badge-sensor", camera: "badge-camera" }[source] || "badge-manual";
    el.innerHTML = `
        <div>
            <div class="history-disease">${disease}</div>
            <div class="history-meta">${item.time}</div>
        </div>
        <span class="history-source-badge ${badgeClass}">${source}</span>
    `;
    list.insertBefore(el, list.firstChild);

    // Keep max 8 history items
    const items = list.querySelectorAll(".history-item");
    if (items.length > 8) items[items.length - 1].remove();
}

function showLoading(text = "Analyzing...") {
    DOM.loadingText.textContent = text;
    DOM.loadingOverlay.style.display = "flex";
}

function hideLoading() {
    DOM.loadingOverlay.style.display = "none";
}

function showToast(msg, type = "info") {
    const existing = document.querySelector(".syndra-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "syndra-toast";
    toast.textContent = msg;
    const colors = { success: "var(--green)", error: "var(--red)", info: "var(--accent2)" };
    toast.style.cssText = `
        position:fixed; bottom:90px; left:50%; transform:translateX(-50%);
        background:var(--bg3); border:1px solid ${colors[type] || colors.info};
        color:var(--text-bright); font-family:var(--font-mono); font-size:0.78rem;
        padding:10px 20px; border-radius:8px; z-index:9999;
        box-shadow:0 4px 20px rgba(0,0,0,0.4);
        animation:fadeIn 0.2s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => { if (toast.parentNode) toast.remove(); }, 3000);
}

function initChatbot() {
    // Toggle panel
    DOM.chatbotFab.addEventListener("click", () => {
        DOM.chatbotPanel.classList.toggle("open");
        DOM.fabBadge.style.display = "none";
    });

    DOM.chatClose.addEventListener("click", () => {
        DOM.chatbotPanel.classList.remove("open");
    });

    // Send message
    DOM.chatSend.addEventListener("click", sendChatMessage);
    DOM.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

async function sendChatMessage() {
    const msg = DOM.chatInput.value.trim();
    if (!msg) return;

    DOM.chatInput.value = "";
    appendChatMsg(msg, "user");
    STATE.chatHistory.push({ role: "user", content: msg });

    // Show typing indicator
    const typingEl = showTypingIndicator();

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: msg,
                history: STATE.chatHistory.slice(-10)
            })
        });
        const data = await resp.json();
        typingEl.remove();

        if (resp.ok && data.response) {
            appendChatMsg(data.response, "bot", data.accuracy);
            STATE.chatHistory.push({ role: "assistant", content: data.response });

            // If chatbot found symptoms, show badge
            if (data.mentioned_symptoms && data.mentioned_symptoms.length > 0 && !DOM.chatbotPanel.classList.contains("open")) {
                DOM.fabBadge.style.display = "flex";
            }
        } else {
            appendChatMsg(data.error || "Sorry, I couldn't process that. Please try again.", "bot");
        }
    } catch (err) {
        typingEl.remove();
        appendChatMsg("Connection error. Please check your network.", "bot");
    }
}

function appendChatMsg(text, role, accuracy) {
    const msg = document.createElement("div");
    msg.className = `chat-msg ${role}`;

    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Convert markdown-like bold **text** to <strong>
    const formatted = text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");

    const accuracyMeta = role === "bot" && Number.isFinite(Number(accuracy))
        ? `<div class="msg-accuracy">Accuracy: ${accuracy}%</div>`
        : "";

    msg.innerHTML = `
        <div class="msg-bubble">${formatted}</div>
        ${accuracyMeta}
        <div class="msg-time">${now}</div>
    `;
    DOM.chatMessages.appendChild(msg);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    return msg;
}

function showTypingIndicator() {
    const typing = document.createElement("div");
    typing.className = "chat-msg bot chat-typing";
    typing.innerHTML = `
        <div class="msg-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    DOM.chatMessages.appendChild(typing);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    return typing;
}

const _originalFetch = window.fetch.bind(window);
window.fetch = function (url, options = {}) {
    if (typeof url === "string" && url.startsWith("/api/")) {
        url = "http://127.0.0.1:5000" + url;
    }
    return _originalFetch(url, options);
};

