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

# ============================================================
# BU DOSYA NE YAPIYOR? (Genel Bakış)
# ============================================================
# 1) Config: klasör yolları, hiperparametreler vs. tanımlanıyor.
# 2) set_seed: her seferinde aynı sonuçları almak için seed ayarlanıyor.
# 3) TinyBaselineCNN: çok küçük bir CNN (baseline model) tanımlanıyor.
# 4) SimpleCNN: asıl kullanmak istediğimiz daha derin CNN tanımlanıyor.
# 5) Device seçimi: GPU / MPS / CPU'dan hangisi varsa otomatik seçiliyor.
# 6) DogEmotionDataset: labels.csv dosyasını okuyup resim + label eşleşmesini yapan custom Dataset sınıfı.
# 7) get_dataloaders: dataset'i train/val/test olarak bölen ve DataLoader dönen fonksiyon.
# 8) train_one_epoch / evaluate: eğitim ve değerlendirme fonksiyonları.
# 9) quick_lr_search: farklı learning rate'leri hızlıca denemek için küçük bir loop.
# 10) main: tüm pipeline'ın çalıştığı yer.
#     - baseline modeli eğit
#     - hızlı LR araması yap
#     - asıl modeli eğit (birden fazla epoch)
#     - en iyi modeli yükle ve test setinde performans ölç
#     - loss/accuracy grafikleri + classification report + confusion matrix kaydet


# ==============================
# 0. Config
# ==============================
# Veri ve çıktı klasörleri + eğitim hiperparametreleri
DATA_DIR = "data/Dog Emotion"       # labels.csv + görüntülerin bulunduğu ana klasör
LABELS_CSV = "labels.csv"           # DATA_DIR altında
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization (ağırlıkların çok büyümesini engellemek için)


# ==============================
# 1. Reproducibility
# ==============================
# Aynı seed ile çalıştırınca aynı sonuçları almak için
def set_seed(seed: int = 42):
    random.seed(seed)              # Python random
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Tüm GPU'lar için seed
    if hasattr(torch.backends, "cudnn"):
        # CUDNN deterministik moda alınıyor
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)  # Script başında 1 kez çağırıyoruz ki her şey sabit olsun


# ==============================
# 2. Baseline model (Tiny CNN)
# ==============================
class TinyBaselineCNN(nn.Module):
    """
    Çok basit, 2 convolution katmanlı baseline model.
    Sadece karşılaştırma için kullanılıyor.
    """
    def __init__(self, num_classes):
        super().__init__()
        # features: 2 convolution bloğu + pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global pooling yerine sabit boyut için AdaptiveAvgPool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)          # Konvolüsyon + pooling
        x = self.avgpool(x)           # Uzaysal boyutu 1x1'e indir
        x = x.view(x.size(0), -1)     # Flatten
        return self.fc(x)             # Sınıflandırma


# ==============================
# 3. Ana model
# ==============================
class SimpleCNN(nn.Module):
    # Asıl kullandığımız daha derin CNN.
    # 4 tane convolution bloğu + BatchNorm + ReLU + MaxPool
    # Sonunda global average pooling ve tek bir Linear katman var.
    def __init__(self, num_classes: int = 4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

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

        # Dropout: overfit'i azaltmak için
        self.dropout = nn.Dropout(p=0.5)
        # AdaptiveAvgPool: giriş boyutu değişse bile 1x1'e çeker
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Son sınıflandırma katmanı
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)          # Feature extractor kısım
        x = self.avgpool(x)           # 8x8 -> 1x1
        x = torch.flatten(x, 1)       # [B, 256, 1, 1] -> [B, 256]
        x = self.dropout(x)           # Dropout
        x = self.classifier(x)        # Logits
        return x


# ==============================
# 4. Device seçimi
# ==============================
# Apple Silicon varsa MPS, yoksa CUDA, o da yoksa CPU kullan.
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
    # Bu sınıf labels.csv'yi okuyup:
    # - her satır için image path + label alıyor
    # - resmi yüklüyor, gerekirse transform uyguluyor
    # - (image, label) döndürüyor
    def __init__(self, csv_path, root_dir, transform=None):
        """
        csv_path: labels.csv yolu
        root_dir: resimlerin bulunduğu ana klasör (csv içindeki path'ler buna göre relative olmalı)
        """
        self.df = pd.read_csv(csv_path)

        # 'label' sütunu yoksa hata ver
        if "label" not in self.df.columns:
            raise ValueError("labels.csv içinde 'label' isimli bir sütun bekleniyor.")

        # Görüntü ismi hangi sütunda? 'filename' ya da 'image' olabilir
        if "filename" in self.df.columns:
            path_col = "filename"
        elif "image" in self.df.columns:
            path_col = "image"
        else:
            raise ValueError("labels.csv içinde görüntü ismi için 'filename' veya 'image' sütunu bulunamadı.")

        # Tam path: root_dir + dosya adı
        self.img_paths = self.df[path_col].apply(lambda x: os.path.join(root_dir, x)).values
        # String label'lar (örneğin "happy", "angry", vs.)
        self.labels_str = self.df["label"].values

        # Label'ları index'e çevir (ör: {"angry":0, "happy":1, ...})
        unique_labels = sorted(self.df["label"].unique())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

        # String label -> int label array
        self.labels = np.array([self.label_to_idx[s] for s in self.labels_str], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        # Toplam örnek sayısı
        return len(self.img_paths)

    def __getitem__(self, idx):
        # idx'inci resmi ve label'ı döndür
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Resmi oku, RGB'ye çevir (grayscale vs. sorun olmasın diye)
        image = Image.open(img_path).convert("RGB")

        # Eğer transform tanımlıysa uygula (resize, normalize vb.)
        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# 6. Dataloaders
# ==============================
def get_dataloaders():
    # labels.csv tam path
    csv_path = os.path.join(DATA_DIR, LABELS_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv bulunamadı: {csv_path}")

    # Train için augmentation içeren transform
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),                  # rastgele yatay çevirme
        transforms.RandomRotation(10),                      # hafif döndürme
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # parlaklık/kontrast oynaması
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Val/Test için augmentation yok, sadece resize + normalize
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Tüm dataset (henüz split edilmemiş)
    full_dataset = DogEmotionDataset(csv_path=csv_path, root_dir=DATA_DIR, transform=None)

    # 70% train, 15% val, 15% test şeklinde oranla
    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # random_split ile 3 parçaya böl (aynı split için seed veriyoruz)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )

    # Her parçaya uygun transform'u set et
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # DataLoader'lar (train shuffle=True, val/test shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # index -> class adı mapping (raporlama için lazım)
    idx_to_class = full_dataset.idx_to_label

    return train_loader, val_loader, test_loader, idx_to_class


# ==============================
# 7. Train / Eval fonksiyonları
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    # Bir epoch boyunca train loader üzerinde eğitimi gerçekleştirir
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm: progress bar için
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()            # Önceki gradient'leri sıfırla
        outputs = model(images)          # Forward
        loss = criterion(outputs, labels)  # Loss hesapla

        loss.backward()                  # Backward
        optimizer.step()                 # Ağırlıkları güncelle

        # epoch toplam loss/acc için biriktir
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)  # En büyük logit hangi sınıf?
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Ortalama loss ve accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, desc="Val"):
    # Train modundan çıkıp evaluation moduna alıyoruz
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Evaluation'da gradient gerekmediği için no_grad
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ==============================
# 8. Quick LR Search (hyperparam deneyi)
# ==============================
def quick_lr_search(train_loader, val_loader, num_classes):
    """
    Çok hızlı LR araması.
    Her LR için 1 epoch train + val -> lr_search.txt'e yazılır.
    """
    # Denenecek learning rate'ler
    lrs = [1e-2, 5e-3, 1e-3, 5e-4]
    results = []

    print("\n=== Quick LR Search ===")
    for lr in lrs:
        print(f"\nTesting LR = {lr}")

        # Her LR için yeni bir model yaratıyoruz (sıfırdan)
        model = SimpleCNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Sadece 1 epoch train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 1 epoch sonrası validation performansı
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, desc=f"Val LR={lr}"
        )

        print(f"LR={lr} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
        results.append((lr, train_acc, val_acc))

    # Sonuçları txt'ye yaz
    lr_search_path = os.path.join(OUTPUT_DIR, "lr_search.txt")
    with open(lr_search_path, "w") as f:
        for lr, tr, val in results:
            f.write(f"LR={lr} -> Train Acc={tr:.4f}, Val Acc={val:.4f}\n")

    print(f"LR search results saved to: {lr_search_path}")


# ==============================
# 9. Training & Testing
# ==============================
def main():
    # Çıktı klasörünü oluştur (yoksa)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dataloader'ları al ve sınıf isimlerini çıkar
    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders()
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print("Classes:", class_names)

    # ============================
    # Baseline Model Quick Test
    # ============================
    print("\n=== Baseline Model Training (Tiny CNN) ===")
    baseline_model = TinyBaselineCNN(num_classes).to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
    baseline_criterion = nn.CrossEntropyLoss()

    # Baseline model için 1 epoch train
    base_train_loss, base_train_acc = train_one_epoch(
        baseline_model, train_loader, baseline_criterion, baseline_optimizer, device
    )

    # Ve validation
    base_val_loss, base_val_acc = evaluate(
        baseline_model, val_loader, baseline_criterion, device, desc="Baseline Val"
    )

    print(f"Baseline Train Acc: {base_train_acc:.4f}")
    print(f"Baseline   Val Acc: {base_val_acc:.4f}")

    # Baseline sonuçlarını txt dosyasına kaydet
    baseline_path = os.path.join(OUTPUT_DIR, "baseline_results.txt")
    with open(baseline_path, "w") as f:
        f.write(f"Baseline Train Acc: {base_train_acc:.4f}\n")
        f.write(f"Baseline Val   Acc: {base_val_acc:.4f}\n")

    print(f"Baseline results saved to: {baseline_path}\n")

    # ============================
    # Quick Learning Rate Search
    # ============================
    # Farklı learning rate'leri hızlıca deniyoruz
    quick_lr_search(train_loader, val_loader, num_classes)

    # ============================
    # Asıl modelin eğitimi
    # ============================
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")  # En iyi (en düşük) validation loss'u tutmak için

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Birden fazla epoch boyunca eğitim
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 30)

        # Train ve Val
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")

        # Epoch sonuçlarını kaydet
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Validation loss daha iyiyse modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Best model updated! Saved to {MODEL_PATH}")

    # ---- Loss curves ----
    # Eğitim/validation loss grafiğini çiz ve kaydet
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curves saved to: {loss_plot_path}")

    # ---- Accuracy curves ----
    # Eğitim/validation accuracy grafiğini çiz ve kaydet
    plt.figure()
    plt.plot(train_accuracies, label="Train acc")
    plt.plot(val_accuracies, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train vs Validation Accuracy")
    plt.tight_layout()
    acc_plot_path = os.path.join(OUTPUT_DIR, "accuracy_curves.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy curves saved to: {acc_plot_path}")

    # ---- Test aşaması ----
    # En iyi model checkpoint'i varsa yükle
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading best model from: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: best model not found, using last epoch weights.")

    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    # Test seti üzerinde ileri yayılım, gradient yok
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Rapor ve confusion matrix için gerçek ve tahmin label'ları sakla
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    test_loss = test_loss / total
    test_acc = correct / total

    # Test sonuçlarını yazdır
    print("\n" + "=" * 20 + " Test Results " + "=" * 20)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}\n")

    # Ayrıntılı classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix (karışıklık matrisi) hesapla ve kaydet
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Dog Emotion")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_path}")

    print("\nAll results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
