import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

import torchvision.models as models                                                                 # MODELI IYILESTIRMEK ICIN EKLEDIM    

from gpu_test import device               

# --- 1. AYARLAR (HYPERPARAMETERS) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4                                                                                # 0.001'den 0.0001'e düşürdüM (Daha hassas öğrenme)
EPOCHS = 20                                                                                         # Biraz daha uzun sürebilir, ezberlemesi zorlaştı çünkü
IMG_SIZE = 224                                                                                      # MODELI IYILESTIRMEK ICIN 224 YAPTIM                                                                                      

# --- 2. VERI HAZIRLIGI (GÜÇLENDİRİLMİŞ) ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                                                                  # ResNet için standart boyut
    # Rastgele kırpma ve yeniden boyutlandırma (En etkili yöntemlerden biri)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),                                            # OVERFITTINGI AZALTMAK ICIN EKLEDIM
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # Renklerle oyna (Model rengi ezberlemesin, şekle odaklansın)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),                  # RENK OYUNLARI EKLEDIM
    transforms.ToTensor(),
    # ImageNet standartlarına göre normalize et (Transfer learning için şart)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                              # IMAGENET NORMALIZASYONU EKLEDIM
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Veriyi yukle
data_path = './dataset'                                                                             

if not os.path.exists(data_path):
    print("HATA: 'dataset' klasoru bulunamadi!")
    exit()

full_dataset = datasets.ImageFolder(data_path, transform=train_transforms)                         
print(f"Toplam resim sayisi: {len(full_dataset)}")
print(f"Siniflar: {full_dataset.classes}")

train_size = int(0.7 * len(full_dataset))                                                           
val_size = int(0.15 * len(full_dataset))                                                            
test_size = len(full_dataset) - train_size - val_size                                              

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])     

# Validation ve Test icin transformlari duzelt
val_data.dataset.transform = val_test_transforms
test_data.dataset.transform = val_test_transforms

# DataLoaderlari olustur
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODEL MIMARISI (TRANSFER LEARNING) ---
def get_resnet_model(num_classes):                                                                  # CNN YERINE TRANSFER LEARNING KULLANIYORUM
    # 1. ResNet18'i indir (Weights=Default diyerek eğitilmiş halini alıyoruz)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. ResNet'in son katmanını (fc) bizim 4 sınıfımıza göre değiştiriyoruz
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Overfitting engellemek için
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# Modeli baslat
num_classes = len(full_dataset.classes)
model = get_resnet_model(num_classes).to(device)                                                    # MODELI BASLATIYORUM

# Optimizer ve Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)                     # WEIGHT DECAY EKLEDIM
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # Loss düşmezse LR i 0.1 ile çarpıp küçültür

# --- 4. EGITIM DONGUSU ---
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Egitim basliyor...")

for epoch in range(EPOCHS):
    # Training
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
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Validation
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
            
    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)                                                                        # LR scheduler'ı epoch sonunda güncelle 
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

print("Egitim tamamlandi!")

# --- 5. GRAFIKLER ---
plt.figure(figsize=(12, 4))

# Loss Grafigi
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Grafigi')
plt.legend()

# Accuracy Grafigi
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Validation Acc')
plt.title('Accuracy Grafigi')
plt.legend()

plt.show()

torch.save(model.state_dict(), 'dog_emotion_model.pth')
print("Model kaydedildi.")