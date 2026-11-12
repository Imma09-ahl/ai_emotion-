"""
Serveur Flask — API de reconnaissance des expressions faciales.
Routes:
  GET  /          → Sert l'interface web principale
  POST /analyze   → Analyse une ou plusieurs images, retourne les émotions en JSON
  POST /capture   → Analyse une image capturée via la webcam (base64)
"""

import sys
import io
import base64
import json
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.model import EmotionCNN

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 Mo max par requête

EMOTION_CONFIG = {
    "angry":    {"fr": "Colère",    "color": "#a53e2a"},
    "disgust":  {"fr": "Dégoût",    "color": "#7a4a1e"},
    "fear":     {"fr": "Peur",      "color": "#c98a42"},
    "happy":    {"fr": "Joie",      "color": "#6b7a3d"},
    "neutral":  {"fr": "Neutre",    "color": "#5a554e"},
    "sad":      {"fr": "Tristesse", "color": "#4a6a8a"},
    "surprise": {"fr": "Surprise",  "color": "#c96442"},
}

# Chargement du modèle au démarrage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = base_dir / "models" / "best_emotion_model.pth"
yunet_path = base_dir / "models" / "face_detection_yunet.onnx"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
CLASSES = checkpoint["classes"]

model = EmotionCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def detect_faces(img_bgr):
    """Détecte les visages via YuNet et retourne la liste des régions [x, y, w, h]."""
    h, w = img_bgr.shape[:2]
    # Résolution de détection adaptée
    det_size = (max(320, w), max(320, h))
    detector = cv2.FaceDetectorYN_create(
        str(yunet_path), "", det_size, score_threshold=0.5, nms_threshold=0.3
    )
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    if faces is None:
        return []
    regions = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        if x2 - x > 10 and y2 - y > 10:
            regions.append({"x": x, "y": y, "w": x2 - x, "h": y2 - y})
    return regions


def predict_face(face_bgr):
    """Prend un crop de visage (BGR), retourne la prédiction et toutes les probabilités."""
    pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    emotion_en = CLASSES[pred_idx]
    cfg = EMOTION_CONFIG.get(emotion_en, {"fr": emotion_en, "color": "#5a554e"})

    all_probs = {
        CLASSES[i]: {
            "fr": EMOTION_CONFIG.get(CLASSES[i], {"fr": CLASSES[i]})["fr"],
            "pct": round(probs[i].item() * 100, 1),
            "color": EMOTION_CONFIG.get(CLASSES[i], {"color": "#5a554e"})["color"]
        }
        for i in range(len(CLASSES))
    }

    return {
        "emotion_en": emotion_en,
        "emotion_fr": cfg["fr"],
        "confidence": round(probs[pred_idx].item() * 100, 1),
        "color": cfg["color"],
        "all_probabilities": all_probs
    }


def analyze_image_bytes(img_bytes):
    """
    Analyse complète d'une image : détection de tous les visages + prédiction par visage.
    Retourne un dict avec les visages détectés et leurs émotions.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Image invalide ou format non supporté."}

    h, w = img_bgr.shape[:2]
    faces = detect_faces(img_bgr)

    if not faces:
        # Pas de visage détecté : on analyse quand même l'image entière
        result = predict_face(img_bgr)
        return {
            "width": w, "height": h,
            "faces_detected": 0,
            "full_image_prediction": result,
            "faces": []
        }

    face_results = []
    for region in faces:
        x, y, fw, fh = region["x"], region["y"], region["w"], region["h"]
        face_crop = img_bgr[y:y+fh, x:x+fw]
        pred = predict_face(face_crop)
        face_results.append({**region, **pred})

    return {
        "width": w, "height": h,
        "faces_detected": len(face_results),
        "faces": face_results
    }


@app.route("/")
def index():
    return render_template("index.html",
                           model_accuracy=round(checkpoint.get("test_acc", 0), 1),
                           num_classes=len(CLASSES))


@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Aucune image reçue."}), 400

    results = []
    for f in files:
        if f.filename == "":
            continue
        img_bytes = f.read()
        analysis = analyze_image_bytes(img_bytes)
        results.append({"filename": f.filename, **analysis})

    return jsonify(results)


@app.route("/capture", methods=["POST"])
def capture():
    """Reçoit une image base64 capturée par la webcam dans le navigateur."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Données manquantes."}), 400

    # Format attendu: "data:image/jpeg;base64,/9j/..."
    b64_data = data["image"].split(",")[-1]
    img_bytes = base64.b64decode(b64_data)
    analysis = analyze_image_bytes(img_bytes)
    return jsonify([{"filename": "capture.jpg", **analysis}])


if __name__ == "__main__":
    print(f"[+] Serveur démarré sur http://127.0.0.1:8000")
    print(f"[+] Modèle EmotionCNN chargé — Précision test: {checkpoint.get('test_acc', 0):.1f}%")
    app.run(host="127.0.0.1", port=8000, debug=False)

