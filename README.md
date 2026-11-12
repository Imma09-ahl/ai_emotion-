# Reconnaissance des Expressions Faciales (Emotion Recognition)

Ce projet a pour objectif de concevoir et développer pas à pas un modèle de Machine Learning / Deep Learning capable d'identifier et de classifier les émotions humaines à partir d'expressions faciales (joie, tristesse, colère, surprise, etc.).

## 📁 Structure du projet

- **`data/raw/`** : Répertoire accueillant le jeu de données FER-2013 (`train/` et `test/` organisés par émotion).
- **`src/`** : Code source modulaire du projet :
  - `download_data.py` : Script d'acquisition et d'extraction automatique du dataset.
  - `dataset.py` : Pipeline PyTorch de chargement des données, normalisation et Data Augmentation.
  - `model.py` : Architecture du réseau de neurones convolutif (CNN).
  - `train.py` : Script d'entraînement, suivi de la perte/précision et sauvegarde du modèle.
  - `evaluate.py` : Script d'évaluation (matrice de confusion, rapport de classification).
- **`models/`** : Répertoire de stockage des poids (`best_emotion_model.pth`), du détecteur facial YuNet (`face_detection_yunet.onnx`), et des visualisations (`confusion_matrix.png`).
- **`app.py`** : Application en temps réel de détection des expressions faciales (webcam, image ou démo).
- **`requirements.txt`** : Dépendances Python nécessaires au projet (PyTorch, OpenCV, etc.).
- **`.gitignore`** : Liste des fichiers et dossiers à exclure du suivi de version Git.
- **`README.md`** : Présentation générale et documentation du projet.

## 🚀 Utilisation du projet

### 1. Lancer l'entraînement manuellement
```bash
python src/train.py 5
```
*(Remplace `5` par le nombre d'époques souhaité. Le meilleur modèle est automatiquement sauvegardé dans `models/best_emotion_model.pth`).*

### 2. Évaluer le modèle et générer la matrice de confusion
```bash
python src/evaluate.py
```
*(Génère le rapport de classification et l'image `models/confusion_matrix.png`).*

### 3. Lancer l'application de détection en temps réel
- **Sur la webcam en direct** :
  ```bash
  python app.py
  ```
  *(Appuyez sur `q` pour fermer la fenêtre vidéo).*
- **Sur une image photo** :
  ```bash
  python app.py --image chemin/vers/photo.jpg
  ```
- **Mode démonstration automatique** :
  ```bash
  python app.py --demo
  ```

## 📊 Jeu de données (FER-2013)

Le dataset contient **35 887 images** en niveaux de gris (48x48 pixels) réparties en 7 classes d'émotions :
- **Colère** (*angry*) : 4 953 images
- **Dégoût** (*disgust*) : 547 images
- **Peur** (*fear*) : 5 121 images
- **Joie** (*happy*) : 8 989 images
- **Neutre** (*neutral*) : 6 198 images
- **Tristesse** (*sad*) : 6 077 images
- **Surprise** (*surprise*) : 4 002 images


