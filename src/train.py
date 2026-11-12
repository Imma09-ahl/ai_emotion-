"""
Script d'entraînement du modèle EmotionCNN.
Gère:
  1. La boucle d'apprentissage (Forward, Loss, Backward, Optimizer).
  2. L'évaluation à chaque époque sur le jeu de test.
  3. La sauvegarde automatique du meilleur modèle dans `models/best_emotion_model.pth`.
  4. La génération des courbes d'apprentissage (perte & précision).
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Ajout du dossier parent au path pour importer les modules de src
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.dataset import get_dataloaders
from src.model import EmotionCNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle sur une époque complète (tous les batches du train).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # 1. Remise à zéro des gradients calculés lors du batch précédent
        optimizer.zero_grad()

        # 2. Passage avant (Forward pass) : prédiction brute
        outputs = model(images)

        # 3. Calcul de l'erreur (Loss) par rapport aux vraies étiquettes
        loss = criterion(outputs, labels)

        # 4. Rétropropagation (Backward pass) : calcul de la dérivée de l'erreur
        loss.backward()

        # 5. Ajustement des poids des neurones (Descente de gradient)
        optimizer.step()

        # Statistiques
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Affichage régulier de la progression
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            current_loss = running_loss / total
            current_acc = (correct / total) * 100
            print(f"    -> Batch {batch_idx+1:03d}/{len(dataloader):03d} | Perte: {current_loss:.4f} | Précision: {current_acc:.2f}%", end="\r", flush=True)

    print()
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Évalue le modèle sur le jeu de test (sans calculer de gradients).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = (correct / total) * 100
    return test_loss, test_acc


def plot_history(history, save_path):
    """
    Génère et sauvegarde les courbes de perte (Loss) et de précision (Accuracy).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))

    # Graphique 1 : Perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["test_loss"], "r-o", label="Test Loss")
    plt.title("Évolution de l'Erreur (Loss)")
    plt.xlabel("Époque")
    plt.ylabel("Perte (Cross Entropy)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Graphique 2 : Précision (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], "r-o", label="Test Accuracy")
    plt.title("Évolution de la Précision (%)")
    plt.xlabel("Époque")
    plt.ylabel("Précision (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Graphique de suivi d'entraînement sauvegardé dans: {save_path}")


def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    # Optimisation des threads CPU pour accélérer l'entraînement
    if hasattr(os, "cpu_count") and os.cpu_count():
        torch.set_num_threads(os.cpu_count())

    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_emotion_model.pth"
    history_plot_path = models_dir / "training_history.png"

    # Détection de l'accélérateur (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Entraînement exécuté sur : {device.type.upper()}")

    # 1. Chargement des données
    train_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)
    print(f"[+] Données chargées : 7 émotions, {len(train_loader.dataset)} images train, {len(test_loader.dataset)} images test.")

    # 2. Instanciation du modèle, de la perte et de l'optimiseur
    model = EmotionCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }

    best_test_acc = 0.0

    print("\n" + "=" * 70)
    print(f"  DÉMARRAGE DE L'ENTRAÎNEMENT ({epochs} ÉPOQUES)")
    print("=" * 70)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[Époque {epoch:02d}/{epochs:02d}] (Taux d'apprentissage : {optimizer.param_groups[0]['lr']:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Ajuste le taux d'apprentissage si la précision stagne
        scheduler.step(test_acc)

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"    Train -> Perte: {train_loss:.4f} | Précision: {train_acc:.2f}%")
        print(f"    Test  -> Perte: {test_loss:.4f} | Précision: {test_acc:.2f}% | Durée: {epoch_duration:.1f}s")

        # Sauvegarde automatique du meilleur modèle
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "classes": classes
            }, best_model_path)
            print(f"    ⭐ Nouveau record de précision ({test_acc:.2f}%) ! Modèle sauvegardé.")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"  ENTRAÎNEMENT TERMINÉ EN {total_time/60:.1f} MINUTES")
    print(f"  Meilleure précision obtenue sur l'examen (Test) : {best_test_acc:.2f}%")
    print(f"  Poids du modèle sauvegardés dans : {best_model_path}")
    print("=" * 70)

    # Sauvegarde des graphiques de performance
    plot_history(history, history_plot_path)

    return best_test_acc


if __name__ == "__main__":
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    train_model(epochs=num_epochs, batch_size=64)

