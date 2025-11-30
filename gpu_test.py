import torch

# GPU kullanılabilir mi kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Kullanılan Cihaz: {device}")

if torch.cuda.is_available():
    print(f"Ekran Kartı Modeli: {torch.cuda.get_device_name(0)}")
    print("Başarılı! Ekran kartını kullanabilirsin.")
else:
    print("Hata: GPU görülmedi. Sürücüleri veya kurulumu kontrol et.")