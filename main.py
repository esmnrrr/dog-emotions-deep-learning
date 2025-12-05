    # --- BURADAN SONRASI YENİ EKLENECEK KISIMLAR ---

# 6. Dönüşümler (Transforms)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([                                             # data augmentation yapiyorum 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),                                         # Yatay çevirme
    transforms.RandomRotation(10),                                                  # Hafif döndürme
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([                                              # validation ve test için sadece normalize ediyorum                                        
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 7. Veri Yükleme ve Bölme
full_dataset = DogEmotionDataset(csv_path=LABELS_CSV, root_dir=DATA_DIR, transform=None) # veriyi yüklüyorum

# Veriyi bölme
total_size = len(full_dataset)
train_size = int(0.7 * total_size)                                                  # %70 i traine ayiriyorum
val_size = int(0.15 * total_size)                                                   # %15 i valide ayiriyorum
test_size = total_size - train_size - val_size                                      # kalan %15 i teste ayiriyorum   

train_subset, val_subset, test_subset = random_split(                                   
    full_dataset, [train_size, val_size, test_size],                                    
    generator=torch.Generator().manual_seed(42)                                        
)

# Subset'lerin transformlarını atama (Biraz trick gerektirir çünkü subset transform tutmaz)
# Pratik çözüm: Dataset içinde transformu dinamik yapabilirsin veya
# Subset'i sarmalayan ayrı bir sınıf yazabilirsin.
# Şimdilik basitlik adına hepsine test_transforms atayalım (Data Augmentation sonra eklersin)
full_dataset.transform = test_transforms 

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)     
val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}, Test size: {len(test_subset)}")

# 8. Model, Loss ve Optimizer Kurulumu
model = SimpleCNN(num_classes=4).to(device)                                          # sinif sayisi 4 oldugu icin 4 yazdim 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # L2 regularization ekliyorum

# 9. Basit Eğitim Fonksiyonu (İskelet)
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()           # Gradyanları sıfırla
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels) # Hata hesabı
        loss.backward()                 # Backward pass
        optimizer.step()                # Ağırlıkları güncelle
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(loader)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad(): # Gradyan hesaplama yok
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(loader)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# Sonraki adım: Döngüyü NUM_EPOCHS kadar çalıştırıp sonuçları kaydetmek.