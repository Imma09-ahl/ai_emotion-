"""
Application principale de Reconnaissance des Expressions Faciales en Temps Réel.
Fonctionnalités:
  1. Flux vidéo webcam en direct (OpenCV).
  2. Détection faciale haute précision via YuNet (FaceDetectorYN).
  3. Classification de l'expression faciale par notre modèle EmotionCNN.
  4. Affichage des boîtes englobantes, libellés en français et barres de confiance.
  5. Modes alternatifs : analyse d'une image fixe (--image) ou démonstration (--demo).
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from src.model import EmotionCNN

# Dictionnaire de traduction et couleurs associées (format BGR pour OpenCV)
EMOTION_CONFIG = {
    "happy":    {"fr": "Joie",      "color": (50, 205, 50)},    # Vert lime
    "surprise": {"fr": "Surprise",  "color": (255, 215, 0)},    # Jaune d'or
    "neutral":  {"fr": "Neutre",    "color": (200, 200, 200)},  # Blanc cassé
    "sad":      {"fr": "Tristesse", "color": (238, 130, 238)},  # Violet / Mauve
    "fear":     {"fr": "Peur",      "color": (255, 105, 180)},  # Rose fuchsia
    "angry":    {"fr": "Colere",    "color": (0, 0, 255)},      # Rouge vif
    "disgust":  {"fr": "Degout",    "color": (0, 140, 255)}     # Orange
}


class EmotionDetectorApp:
    def __init__(self, model_path=None, yunet_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Chargement du modèle CNN d'émotions
        if model_path is None:
            model_path = base_dir / "models" / "best_emotion_model.pth"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable à {model_path}. Lancez d'abord l'entraînement.")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = checkpoint.get("classes", list(EMOTION_CONFIG.keys()))

        self.model = EmotionCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 2. Pipeline de transformation pour le modèle (Grayscale 48x48, normalisation)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 3. Chargement du détecteur de visage YuNet
        if yunet_path is None:
            yunet_path = base_dir / "models" / "face_detection_yunet.onnx"
        else:
            yunet_path = Path(yunet_path)

        if not yunet_path.exists():
            raise FileNotFoundError(f"Détecteur YuNet introuvable à {yunet_path}.")

        self.face_detector = cv2.FaceDetectorYN_create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3
        )

        print(f"[+] Application initialisée avec succès.")
        print(f"    - Modèle : EmotionCNN ({len(self.classes)} émotions, précision test: {checkpoint.get('test_acc', 0):.1f}%)")
        print(f"    - Détecteur de visage : YuNet ONNX")
        print(f"    - Processeur : {self.device.type.upper()}")

    def predict_emotion(self, face_bgr):
        """
        Prend une région de visage découpée (BGR), et retourne la classe et les probabilités.
        """
        pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        emotion_en = self.classes[pred_idx]
        conf = probs[pred_idx].item() * 100
        return emotion_en, conf, probs.cpu().numpy()

    def process_frame(self, frame):
        """
        Détecte les visages dans l'image et dessine les prédictions d'émotions.
        """
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))

        _, faces = self.face_detector.detect(frame)
        if faces is None:
            return frame

        for face in faces:
            x, y, box_w, box_h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            # Assurer que les coordonnées restent dans les limites de l'image
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + box_w)
            y2 = min(h, y + box_h)

            if x2 <= x or y2 <= y or (x2 - x) < 20 or (y2 - y) < 20:
                continue

            face_roi = frame[y:y2, x:x2]
            emotion_en, conf, _ = self.predict_emotion(face_roi)

            cfg = EMOTION_CONFIG.get(emotion_en, {"fr": emotion_en, "color": (0, 255, 0)})
            label_text = f"{cfg['fr']} : {conf:.0f}%"
            color = cfg["color"]

            # Boîte englobante stylisée
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            # Bandeau de texte supérieur
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, max(0, y - text_h - 10)), (x + text_w + 10, y), color, cv2.FILLED)
            cv2.putText(
                frame,
                label_text,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return frame

    def run_webcam(self, camera_index=0):
        """
        Lance la capture webcam en direct.
        """
        print(f"\n[+] Démarrage de la webcam (index {camera_index})...")
        print("[!] Appuyez sur la touche 'q' dans la fenêtre vidéo pour quitter.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[!] Impossible d'ouvrir la webcam. Lancement du mode démonstration sur images de test...")
            self.run_demo()
            return

        window_name = "Reconnaissance des Émotions Faciales - PyTorch & OpenCV"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Erreur lors de la lecture du flux vidéo.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow(window_name, processed_frame)

            # Quitter si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[+] Application arrêtée proprement.")

    def run_image(self, image_path, save_output=True):
        """
        Analyse une image statique.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"[!] Fichier introuvable : {img_path}")
            return

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[!] Erreur de lecture de l'image : {img_path}")
            return

        processed = self.process_frame(frame)

        if save_output:
            out_path = base_dir / "models" / f"output_{img_path.name}"
            cv2.imwrite(str(out_path), processed)
            print(f"[+] Résultat sauvegardé dans : {out_path}")

        cv2.imshow("Résultat Détection d'Émotions", processed)
        print("[!] Appuyez sur n'importe quelle touche pour fermer la fenêtre...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_demo(self):
        """
        Mode démo : teste automatiquement plusieurs visages du jeu de test.
        """
        print("\n" + "=" * 60)
        print("  DÉMONSTRATION AUTOMATIQUE SUR ÉCHANTILLONS DE TEST")
        print("=" * 60)

        test_dir = base_dir / "data" / "raw" / "test"
        sample_emotions = ["happy", "surprise", "sad", "angry", "neutral"]

        for emo in sample_emotions:
            emo_folder = test_dir / emo
            if not emo_folder.exists():
                continue

            images = list(emo_folder.glob("*.jpg"))
            if not images:
                continue

            sample_img_path = images[0]
            raw_img = cv2.imread(str(sample_img_path))

            # Prédiction directe sur le visage
            emotion_en, conf, _ = self.predict_emotion(raw_img)
            cfg_pred = EMOTION_CONFIG.get(emotion_en, {"fr": emotion_en})
            cfg_real = EMOTION_CONFIG.get(emo, {"fr": emo})

            status = "✅ SUCCÈS" if emotion_en == emo else "⚠️ ERREUR"
            print(f"[{status}] Vraie émotion: {cfg_real['fr']:<10} | Prédite: {cfg_pred['fr']:<10} ({conf:.1f}%)")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Application de reconnaissance des expressions faciales")
    parser.add_argument("--image", type=str, help="Chemin vers une image fixe à analyser")
    parser.add_argument("--demo", action="store_true", help="Lance la démo automatique sur le dataset")
    parser.add_argument("--camera", type=int, default=0, help="Index de la webcam (défaut: 0)")
    args = parser.parse_args()

    app = EmotionDetectorApp()

    if args.demo:
        app.run_demo()
    elif args.image:
        app.run_image(args.image)
    else:
        app.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
