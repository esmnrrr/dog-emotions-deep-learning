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

# 0. Config
DATA_DIR = "data/Dog Emotion"       # labels.csv + görüntülerin bulunduğu ana klasör
LABELS_CSV = "labels.csv"           # DATA_DIR altında
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization

# 1. Reproducibility
def set_seed(seed: int = 42):                                                            # zar atma olayini sabitliyorum
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

# 2. Baseline model (Tiny CNN)
class TinyBaselineCNN(nn.Module):                                                        # veri akisi saglaniyor mu diye kontrol amacli basit model kurdum 
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

# 3. Ana model
class SimpleCNN(nn.Module):                                                                    
    def __init__(self, num_classes: int = 4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),                                 # kenar koseleri tespit ediyorum 
            nn.BatchNorm2d(32),                                                         # trainingde sayıların çok büyümesini veya küçülmesini engelliyorum      
            nn.ReLU(inplace=True),                                                      # non linear lik ekliyorum
            nn.MaxPool2d(2),  # 128 -> 64                                               # resmi yarıya indiriyorum

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

        self.dropout = nn.Dropout(p=0.5)                                                 # overfittingi engellemek icin noronlarin yarisini egitim sirasinda kapatiyorum
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# 4. Device seçimi
from gpu_test import device                                         # gpu_test.py dosyasından device bilgisini alacagim icin ekledim

# 5. Dataset
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

# 6. Veri yükleme ve ön işleme

class SubsetWithTransform(Dataset):

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_dataloaders_fixed():
    csv_path = os.path.join(DATA_DIR, LABELS_CSV)
    
    # Transformlar
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ham dataset (transformsuz yükle)
    # Not: Dataset sınıfındaki __getitem__ içindeki transform satırını kaldırman gerekebilir 
    # ya da None yolladığımız için sorun çıkmaz.
    full_dataset = DogEmotionDataset(csv_path=csv_path, root_dir=DATA_DIR, transform=None)

    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Bölme işlemi
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrapper ile transformları ata
    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset   = SubsetWithTransform(val_subset, transform=val_test_transform)
    test_dataset  = SubsetWithTransform(test_subset, transform=val_test_transform)

    # Loaderlar
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, full_dataset.idx_to_label

# 7. Training & Evaluation Loops

def train_epoch(model, loader, criterion, optimizer, device):
    model.train() # Modeli eğitim moduna al
    running_loss = 0.0
    correct = 0
    total = 0

    # Tqdm ile ilerleme çubuğu
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # 1. Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # İstatistikler
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress bar güncelle
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval() # Modeli değerlendirme moduna al (Dropout kapanır vs.)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Gradyan hesaplama yok (Hafıza tasarrufu)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 8. Main Execution

def main():
    # Klasör oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Veriyi Yükle
    print("Veri yükleniyor...")
    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders_fixed()
    num_classes = len(idx_to_class)
    print(f"Sınıflar: {idx_to_class}")
    print(f"Eğitim: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 2. Modeli Başlat
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning Rate Scheduler (Opsiyonel ama önerilir: Loss düşmezse LR azaltır)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 3. Eğitim Döngüsü
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nEğitim başlıyor... ({NUM_EPOCHS} Epoch)")
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # LR Scheduler güncelle
        scheduler.step(val_loss)

        # Kayıt tut
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved_msg = "-> Model Kaydedildi!"
        else:
            saved_msg = ""

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% {saved_msg}")

    # 4. Sonuçları Görselleştir
    plt.figure(figsize=(12, 5))
    
    # Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
    print(f"\nGrafikler {OUTPUT_DIR} klasörüne kaydedildi.")
    plt.show()

    # 5. Test Seti ile Final Değerlendirme
    print("\nTest seti üzerinde en iyi model değerlendiriliyor...")
    model.load_state_dict(torch.load(MODEL_PATH)) # En iyi ağırlıkları yükle
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Confusion Matrix (Opsiyonel Detay)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(all_labels, all_preds, target_names=list(idx_to_class.values())))

if __name__ == "__main__":
    main()