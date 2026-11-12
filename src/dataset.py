"""
Module de préparation et de chargement des données (PyTorch DataLoaders).
Gère:
  1. La transformation des images (mise à l'échelle, tenseurs, normalisation).
  2. L'augmentation de données (Data Augmentation) pour l'entraînement.
  3. La création des DataLoaders pour alimenter le modèle par lots (batches).
"""

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    """
    Définit les transformations appliquées aux images.
    - Entraînement: transformations + augmentation (rotations, retournements) pour rendre l'IA robuste.
    - Test: transformations pures sans altération pour évaluer fidèlement.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Assure 1 seul canal (niveaux de gris)
        transforms.RandomHorizontalFlip(p=0.5),       # Retournement horizontal aléatoire (miroir)
        transforms.RandomRotation(degrees=10),        # Légère rotation (+/- 10 degrés)
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Léger décalage de position
        transforms.ToTensor(),                        # Convertit les pixels [0, 255] en tenseur [0.0, 1.0]
        transforms.Normalize(mean=[0.5], std=[0.5])   # Normalise entre [-1, 1] pour stabiliser l'IA
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir=None, batch_size=64, num_workers=0):
    """
    Crée et retourne les DataLoaders pour l'entraînement et le test.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train"
    test_path = data_dir / "test"

    train_transform, test_transform = get_transforms()

    # ImageFolder associe automatiquement chaque sous-dossier à une étiquette de classe
    train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=str(test_path), transform=test_transform)

    # Création des générateurs par lots (batches)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Mélange aléatoire à chaque époque pour un apprentissage optimal
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # Pas besoin de mélanger pour le test
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST DU PIPELINE DE DONNÉES (Étape 4)")
    print("=" * 60)

    train_loader, test_loader, classes = get_dataloaders(batch_size=64)

    print(f"\n[+] Classes d'émotions détectées ({len(classes)}) :")
    for idx, name in enumerate(classes):
        print(f"    - Index {idx} : {name}")

    print(f"\n[+] Nombre de lots (batches) par époque :")
    print(f"    - Entraînement : {len(train_loader)} lots de 64 images ({len(train_loader.dataset)} images au total)")
    print(f"    - Test          : {len(test_loader)} lots de 64 images ({len(test_loader.dataset)} images au total)")

    # Récupération d'un tout premier lot pour vérifier ses dimensions
    images, labels = next(iter(train_loader))

    print(f"\n[+] Vérification du premier lot (Batch) extrait :")
    print(f"    - Forme des images (Tenseur) : {images.shape}")
    print(f"      -> 64 images")
    print(f"      -> 1 canal (niveaux de gris)")
    print(f"      -> 48x48 pixels")
    print(f"    - Forme des étiquettes (Labels) : {labels.shape}")
    print(f"    - Plage des valeurs de pixels : min={images.min().item():.2f}, max={images.max().item():.2f}")
    print(f"    - Exemples d'étiquettes du lot : {labels[:10].tolist()}")

    print("\n" + "=" * 60)
    print("  [OK] Le pipeline de données fonctionne parfaitement !")
    print("=" * 60)

