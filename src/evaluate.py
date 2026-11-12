"""
Script d'évaluation du modèle d'émotions.
Gère:
  1. Le chargement du modèle sauvegardé (best_emotion_model.pth).
  2. L'évaluation sur les 7 178 images de test.
  3. Le calcul du rapport de classification (Precision, Recall, F1-Score).
  4. La génération et sauvegarde de la matrice de confusion (models/confusion_matrix.png).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.dataset import get_dataloaders
from src.model import EmotionCNN

EMOTION_FR = {
    "angry": "Colère",
    "disgust": "Dégoût",
    "fear": "Peur",
    "happy": "Joie",
    "neutral": "Neutre",
    "sad": "Tristesse",
    "surprise": "Surprise"
}

def evaluate_model():
    model_path = base_dir / "models" / "best_emotion_model.pth"
    if not model_path.exists():
        print(f"[!] Erreur: aucun modèle trouvé à {model_path}. Entraînez d'abord le modèle.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Évaluation sur : {device.type.upper()}")

    # 1. Chargement des données de test
    _, test_loader, classes = get_dataloaders(batch_size=64)
    classes_fr = [EMOTION_FR.get(c, c) for c in classes]

    # 2. Chargement des poids du modèle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[+] Modèle chargé avec succès (Époque {checkpoint.get('epoch')}, Précision enregistrée: {checkpoint.get('test_acc'):.2f}%).")

    all_preds = []
    all_labels = []

    print("[+] Analyse des 7 178 images de test en cours...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. Calcul de la précision globale
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n{'='*65}")
    print(f"  RÉSULTATS DE L'EXAMEN FINAL : PRÉCISION GLOBALE = {overall_acc:.2f} %")
    print(f"{'='*65}\n")

    # 4. Rapport de classification détaillé
    report = classification_report(all_labels, all_preds, target_names=classes_fr, digits=3)
    print("Détail par émotion (Précision, Rappel, F1-Score) :\n")
    print(report)

    # 5. Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    # Normalisation par ligne (pourcentage de détection par vraie émotion)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matrice de Confusion (% de réussite) - Score global: {overall_acc:.1f}%", fontsize=13, pad=15)
    plt.colorbar(label="% de prédictions")

    tick_marks = np.arange(len(classes_fr))
    plt.xticks(tick_marks, classes_fr, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, classes_fr, fontsize=11)

    # Affichage des valeurs dans chaque case
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"
            color = "white" if cm_norm[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", color=color, fontsize=8)

    plt.ylabel('Vraie émotion (Réalité)', fontsize=12)
    plt.xlabel('Émotion prédite par l\'IA', fontsize=12)
    plt.tight_layout()

    save_path = base_dir / "models" / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Matrice de confusion sauvegardée dans: {save_path}")
    print(f"{'='*65}")

if __name__ == "__main__":
    evaluate_model()

