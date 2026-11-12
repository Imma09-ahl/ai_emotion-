"""
Architecture du modèle d'Intelligence Artificielle pour la reconnaissance d'émotions.
Réseau de neurones convolutif (CNN - Convolutional Neural Network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    Réseau de neurones convolutif à 3 blocs de convolution
    suivis d'un classifieur dense pour 7 émotions.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # --- BLOC CONVOLUTIF 1 (Détection des contours, contrastes simples) ---
        # Entrée: (Batch, 1 canal, 48x48 pixels)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Divise la taille par 2: 48x48 -> 24x24
        self.drop1 = nn.Dropout(0.25)

        # --- BLOC CONVOLUTIF 2 (Détection des formes: yeux, lèvres, sourcils) ---
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24 -> 12x12
        self.drop2 = nn.Dropout(0.25)

        # --- BLOC CONVOLUTIF 3 (Combinaison complexe: expressions faciales globales) ---
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12 -> 6x6
        self.drop3 = nn.Dropout(0.25)

        # --- CLASSIFIEUR DENSE (Prise de décision finale sur les 7 émotions) ---
        # 128 canaux x 6 pixels x 6 pixels = 4608 caractéristiques en entrée
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.50)  # Réduit le surapprentissage
        self.fc2 = nn.Linear(256, num_classes)  # Sortie brute (logits) pour les 7 classes

    def forward(self, x):
        """
        Définit le trajet complet de l'image à travers les couches de neurones.
        """
        # Bloc 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Bloc 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Bloc 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Aplatissement (Flatten) de la grille 2D en vecteur 1D
        x = x.view(x.size(0), -1)

        # Classification dense
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.fc2(x)

        return x

    def predict_proba(self, x):
        """
        Retourne les probabilités (en pourcentage via Softmax) pour chaque classe.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST DU MODÈLE DE VISION (Étape 5)")
    print("=" * 60)

    # Instanciation du modèle
    model = EmotionCNN(num_classes=7)

    # Calcul du nombre de paramètres apprenables (les "poids" que l'IA va ajuster)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[+] Architecture instanciée avec succès.")
    print(f"    - Nombre total de paramètres apprenables : {total_params:,} neurones/connexions")

    # Simulation d'un lot de 4 images 48x48 pour tester le passage avant (forward pass)
    dummy_input = torch.randn(4, 1, 48, 48)
    print(f"\n[+] Envoi d'un lot fictif de test : forme {dummy_input.shape}")

    output_logits = model(dummy_input)
    print(f"    - Forme de sortie brute (logits) : {output_logits.shape} -> [4 images, 7 émotions]")

    probs = model.predict_proba(dummy_input)
    print(f"    - Probabilités calculées pour la 1ère image (somme = {probs[0].sum():.2f}) :")
    for i, p in enumerate(probs[0]):
        print(f"      Classe {i} : {p.item() * 100:.1f} %")

    print("\n" + "=" * 60)
    print("  [OK] Le modèle EmotionCNN est validé et prêt à s'entraîner !")
    print("=" * 60)

