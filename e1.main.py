import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from gpu_test import device                                                                         # gpu_test.py dosyasından device bilgisini alacagim icin ekledim

# --- 1. AYARLAR (HYPERPARAMETERS) ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
IMG_SIZE = 128                                                                                      # Tum resimlerin boyutu (128x128) ayarliyoruz

# --- 2. VERI HAZIRLIGI ---
# Veri artirma (Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                                                        # Resim yuklenirken anlik olarak kucultulur
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Veriyi yukle
data_path = './dataset'                                                                             # Veri setini yukluyrouz 

if not os.path.exists(data_path):
    print("HATA: 'dataset' klasoru bulunamadi!")
    exit()

full_dataset = datasets.ImageFolder(data_path, transform=train_transforms)                          # Tum veriyi okuyoruz
print(f"Toplam resim sayisi: {len(full_dataset)}")
print(f"Siniflar: {full_dataset.classes}")

train_size = int(0.7 * len(full_dataset))                                                           # %70 oraninda veriyi traine ayiriyoruz
val_size = int(0.15 * len(full_dataset))                                                            # %15 oraninda veriyi valide ayiriyoruz
test_size = len(full_dataset) - train_size - val_size                                               # Kalan %15 oraninda veriyi teste ayiriyoruz

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])     # Veriyi sanal olarak boluyoruz

# Validation ve Test icin transformlari duzelt
val_data.dataset.transform = val_test_transforms
test_data.dataset.transform = val_test_transforms

# DataLoaderlari olustur
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODEL MIMARISI (CNN) ---
class DogEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogEmotionCNN, self).__init__()
        
        # 1. Blok
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 2. Blok
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 3. Blok
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dense Katmanlar
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Modeli baslat
num_classes = len(full_dataset.classes)
model = DogEmotionCNN(num_classes).to(device)

# Optimizer ve Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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