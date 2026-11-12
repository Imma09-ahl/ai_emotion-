"""
Script de téléchargement et de structuration du jeu de données FER-2013.
Organise les images au format standard ImageFolder pour PyTorch:
    data/raw/train/<emotion>/img_xxxxx.jpg
    data/raw/test/<emotion>/img_xxxxx.jpg
"""

import os
import io
import pickle
import urllib.request
from pathlib import Path
from collections import Counter
from PIL import Image

# URLs des fichiers source FER-2013 hébergés sur Hugging Face (miroir compressé)
URLS = {
    "train": "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/train.pt",
    "test": "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/test.pt",
}

# Traduction des émotions pour l'affichage pédagogique
EMOTION_TRANSLATION = {
    "angry": "Colère",
    "disgust": "Dégoût",
    "fear": "Peur",
    "happy": "Joie",
    "neutral": "Neutre",
    "sad": "Tristesse",
    "surprise": "Surprise",
}

def download_and_extract(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    stats = {}

    for split, url in URLS.items():
        split_dir = output_dir / split
        print(f"\n[+] Téléchargement de l'ensemble '{split}'...")
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        
        print(f"[+] Décompression de '{split}' ({len(data) / (1024*1024):.1f} Mo)...")
        items = pickle.loads(data)
        
        counter = Counter()
        for idx, item in enumerate(items):
            label = item["labels"]
            img_bytes = item["img_bytes"]
            
            emotion_dir = split_dir / label
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            img_path = emotion_dir / f"{idx:05d}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            counter[label] += 1
            
            if (idx + 1) % 5000 == 0 or (idx + 1) == len(items):
                print(f"    -> {idx + 1}/{len(items)} images extraites...")
        
        stats[split] = counter

    return stats


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    
    print("=" * 60)
    print("  ACQUISITION DU JEU DE DONNÉES FER-2013 (Étape 3)")
    print("=" * 60)
    
    stats = download_and_extract(raw_data_dir)
    
    print("\n" + "=" * 60)
    print("  RÉSUMÉ DU JEU DE DONNÉES")
    print("=" * 60)
    
    emotions = sorted(list(EMOTION_TRANSLATION.keys()))
    
    header = f"{'Émotion':<12} | {'En français':<14} | {'Train':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    
    total_train = 0
    total_test = 0
    
    for emo in emotions:
        train_count = stats.get("train", {}).get(emo, 0)
        test_count = stats.get("test", {}).get(emo, 0)
        total_emo = train_count + test_count
        
        total_train += train_count
        total_test += test_count
        
        print(f"{emo:<12} | {EMOTION_TRANSLATION[emo]:<14} | {train_count:<8} | {test_count:<8} | {total_emo:<8}")
        
    print("-" * len(header))
    print(f"{'TOTAL':<12} | {'':<14} | {total_train:<8} | {total_test:<8} | {total_train + total_test:<8}")
    print("=" * 60)
    print(f"\n[OK] Données stockées avec succès dans: {raw_data_dir}")

if __name__ == "__main__":
    main()

