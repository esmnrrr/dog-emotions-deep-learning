import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================
# 0. Config
# ==============================
DATA_DIR = "data/Dog Emotion"       # labels.csv + görüntülerin bulunduğu ana klasör
LABELS_CSV = "labels.csv"           # DATA_DIR altında
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization


# ==============================
# 1. Reproducibility
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# ==============================
def set_seed(seed: int = 42):
=======
def set_seed(seed: int = 42):                                   # zar atma olayini sabitliyorum
>>>>>>> Stashed changes
=======
def set_seed(seed: int = 42):                                   # zar atma olayini sabitliyorum
>>>>>>> Stashed changes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# ==============================
# 2. Baseline model (Tiny CNN)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# ==============================
class TinyBaselineCNN(nn.Module):
=======
class TinyBaselineCNN(nn.Module):                                   # veri akisi saglaniyor mu diye kontrol amacli basit model kurdum 
>>>>>>> Stashed changes
=======
class TinyBaselineCNN(nn.Module):                                   # veri akisi saglaniyor mu diye kontrol amacli basit model kurdum 
>>>>>>> Stashed changes
    """
    Çok basit, 2 convolution katmanlı baseline model.
    Sadece karşılaştırma için kullanılıyor.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==============================
# 3. Ana model
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),                             # kenar koseleri tespit ediyorum
            nn.BatchNorm2d(32),                                                     # trainingde sayıların çok büyümesini veya küçülmesini engelliyorum
            nn.ReLU(inplace=True),                                                  # non linear lik ekliyorum
            nn.MaxPool2d(2),  # 128 -> 64                                           # resmi yarıya indiriyorum

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        self.dropout = nn.Dropout(p=0.5)                                            # overfittingi engellemek icin noronlarin yarisini egitim sirasinda kapatiyorum
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ==============================
# 4. Device seçimi
# ==============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ==============================
# 5. Dataset
# ==============================
class DogEmotionDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        csv_path: labels.csv yolu
        root_dir: resimlerin bulunduğu ana klasör (csv içindeki path'ler buna göre relative olmalı)
        """
        self.df = pd.read_csv(csv_path)

        if "label" not in self.df.columns:
            raise ValueError("labels.csv içinde 'label' isimli bir sütun bekleniyor.")

        if "filename" in self.df.columns:
            path_col = "filename"
        elif "image" in self.df.columns:
            path_col = "image"
        else:
            raise ValueError("labels.csv içinde görüntü ismi için 'filename' veya 'image' sütunu bulunamadı.")

        self.img_paths = self.df[path_col].apply(lambda x: os.path.join(root_dir, x)).values
        self.labels_str = self.df["label"].values

        unique_labels = sorted(self.df["label"].unique())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

        self.labels = np.array([self.label_to_idx[s] for s in self.labels_str], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):                                                     # veri setinden bir örnek alıyorum
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
<<<<<<< Updated upstream
<<<<<<< Updated upstream


# ==============================
# 6. Dataloaders
# ==============================
def get_dataloaders():
    csv_path = os.path.join(DATA_DIR, LABELS_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv bulunamadı: {csv_path}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = DogEmotionDataset(csv_path=csv_path, root_dir=DATA_DIR, transform=None)

    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    idx_to_class = full_dataset.idx_to_label

    return train_loader, val_loader, test_loader, idx_to_class
=======
    
>>>>>>> Stashed changes
=======
    
>>>>>>> Stashed changes
