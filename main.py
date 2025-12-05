import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from gpu_test import device                                         # gpu_test.py dosyasından device bilgisini alacagim icin ekledim

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5                                                        # Early Stopping için sabır sayısı
WEIGHT_DECAY = 1e-4                                                 # L2 regularization

# 1. Reproducibility 
def set_seed(seed=42):                                              # zar atma olayini sabitliyorum
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 2. Veri Ön İşleme ve Augmentation 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                                  # Resimleri standart boyuta getir
    transforms.RandomHorizontalFlip(),                              # Rastgele yatay çevir (Augmentation)
    transforms.RandomRotation(10),                                  # Rastgele döndür (Augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standart normalizasyon
])

# Validation ve Test için sadece Resize ve Normalize yapılır (Augmentation yapılmaz!)
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataseti Yükleme
data_path = './dataset'                                             # Veri setini yukluyrouz 

if not os.path.exists(data_path):
    print("HATA: 'dataset' klasoru bulunamadi!")
    exit()
    
full_dataset = datasets.ImageFolder(data_path, transform=train_transforms)  # Tum veriyi okuyoruz
print(f"Toplam resim sayisi: {len(full_dataset)}")
print(f"Siniflar: {full_dataset.classes}")

train_size = int(0.7 * len(full_dataset))                                                           # %70 oraninda veriyi traine ayiriyoruz
val_size = int(0.15 * len(full_dataset))                                                            # %15 oraninda veriyi valide ayiriyoruz
test_size = len(full_dataset) - train_size - val_size                                               # Kalan %15 oraninda veriyi teste ayiriyoruz

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])     # Veriyi sanal olarak boluyoruz

# Validation ve Test setlerinin transformlarını düzeltelim (Augmentation'ı kaldıralım)
val_data.dataset.transform = val_test_transforms
test_data.dataset.transform = val_test_transforms

# DataLoader'ları oluşturma
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Mimarisi (CNN)
class DogEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogEmotionCNN, self).__init__()
        
        # 1. Blok
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),                                 # kenar koseleri tespit ediyorum 
            nn.BatchNorm2d(32),                                                         # trainingde sayıların çok büyümesini veya küçülmesini engelliyorum      
            nn.ReLU(inplace=True),                                                      # non linear lik ekliyorum
            nn.MaxPool2d(2, 2),  # 128 -> 64                                            # resmi yarıya indiriyorum
        )
        
        # 2. Blok
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 3. Blok
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully Connected Katmanı
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512), # 224 -> 112 -> 56 -> 28
            nn.ReLU(),
            nn.Dropout(0.5),                                                            # overfittingi engellemek icin noronlarin yarisini egitim sirasinda kapatiyorum
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = DogEmotionCNN(num_classes=len(full_dataset.classes)).to(device)
print(model)

# 4. Optimizasyon ve Loss Fonksiyonu
criterion = nn.CrossEntropyLoss()                                                       # Fully connected için uygun loss fonksiyonu    
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)                            # Adam optimizer kullanıyoruz

# Learning Rate Scheduler: Loss düşmezse LR'yi azaltır
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Eğitim Fonksiyonu
def train_model(model, train_loader, val_loader, epochs, patience):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Early stopping sayacı

    print("Egitim basliyor...")
    
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        
        # --- Validation Loop ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Listelere Kayıt
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # Scheduler Adımı
        scheduler.step(epoch_val_loss)
        
        # Early Stopping Kontrolü
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')                            # En iyi modeli kaydet
            print("  --> Model Kaydedildi (Loss düştü)")
        else:
            epochs_no_improve += 1
            print(f"  --> İyileşme yok. Sabır: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print("Early Stopping tetiklendi! Eğitim durduruluyor.")
            break
            
    return train_losses, val_losses, train_accs, val_accs

# Eğitimi Başlat
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE)
print("Eğitim tamamlandı!")

# 5. Grafikler
plt.figure(figsize=(12, 5))

# Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.show()                                                                              # Grafikleri göster

# En iyi modeli yükleyip Test Seti üzerinde değerlendirme
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix ve Rapor
cm = confusion_matrix(all_labels, all_preds)
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()