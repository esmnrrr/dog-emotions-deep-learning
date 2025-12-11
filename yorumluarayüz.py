# ================================================
# Dog Emotion Classifier · Flask Web Uygulaması
# ================================================
# Bu dosya, daha önce eğittiğimiz köpek duygu sınıflandırma modelini
# web üzerinden kullanılabilir hale getirir:
#   - SimpleCNN mimarisini yeniden tanımlar.
#   - Eğitilmiş ağırlıkları `outputs/best_model.pt` dosyasından yükler.
#   - Yüklenen resmi train sırasında kullanılan transform'lar ile işler.
#   - Flask + HTML/CSS ile şık bir arayüz üzerinden tahmin sonucunu gösterir.
#   - Tüm kod korunmuş, sadece açıklama amaçlı yorum satırları eklenmiştir.

import os
from io import BytesIO

from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import base64

# ==============================
# 1. Config
# ==============================
DATA_DIR = "data/Dog Emotion"
LABELS_CSV = "labels.csv"
MODEL_PATH = "outputs/best_model.pt"
IMAGE_SIZE = 128


# ==============================
# 2. Model (train ile aynı)
# ==============================
# Basit ama güçlü bir CNN mimarisi: eğitim scriptindeki SimpleCNN ile birebir aynı,
# sadece inference (tahmin) tarafında modeli tekrar tanımlıyoruz.

class SimpleCNN(nn.Module):
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

        self.dropout = nn.Dropout(p=0.5)
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
# 3. Device
# ==============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device for inference:", device)


# ==============================
# 4. Class mapping
# ==============================
# labels.csv dosyasını okuyup, oradaki benzersiz duygu etiketlerini index'e çeviren yardımcı fonksiyon.
# Böylece modelin çıktısındaki sınıf indeksini tekrar string isme çevirebiliyoruz.

def load_class_mapping():
    csv_path = os.path.join(DATA_DIR, LABELS_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels.csv bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)
    classes = sorted(df["label"].unique())
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return idx_to_class


idx_to_class = load_class_mapping()
num_classes = len(idx_to_class)
class_names = [idx_to_class[i] for i in range(num_classes)]


# ==============================
# 5. Model yükleme (inference için)
# ==============================
model = SimpleCNN(num_classes=num_classes)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ==============================
# 6. Transform (inference ile tutarlı)
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Yüklenen tek bir PIL Image için:
#   1) RGB'ye çevirir
#   2) Aynı train pipeline'ındaki transform'u uygular
#   3) Modelden geçirdikten sonra softmax ile olasılıkları hesaplar
#   4) Tahmin edilen sınıfı, o sınıfın olasılığını ve tüm sınıfların olasılık listesini döner.
def predict_image(pil_image: Image.Image):
    pil_image = pil_image.convert("RGB")
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    result = []
    for i, p in enumerate(probs):
        result.append((class_names[i], float(p)))

    top_idx = int(probs.argmax())
    top_class = class_names[top_idx]
    top_prob = float(probs[top_idx])

    result_sorted = sorted(result, key=lambda x: x[1], reverse=True)
    return top_class, top_prob, result_sorted


# ==============================
# 7. Flask app + gelişmiş tasarım
# ==============================
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="tr" data-theme="dark">
<head>
    <meta charset="utf-8">
    <title>Dog Emotion Classifier · Inference UI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg: #020617;
            --bg-soft: #020617;
            --card: linear-gradient(145deg,#020617,#020617);
            --card-soft: #020617;
            --accent: #3b82f6;
            --accent-soft: rgba(59,130,246,0.12);
            --accent-strong: #22c55e;
            --text: #f9fafb;
            --muted: #9ca3af;
            --border: #1f2937;
            --danger: #f97373;
        }
        :root[data-theme="light"] {
            --bg: #f3f4f6;
            --bg-soft: #e5e7eb;
            --card: #ffffff;
            --card-soft: #ffffff;
            --accent: #2563eb;
            --accent-soft: rgba(37,99,235,0.08);
            --accent-strong: #16a34a;
            --text: #0f172a;
            --muted: #6b7280;
            --border: #d1d5db;
            --danger: #dc2626;
        }

        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                        "SF Pro Text", "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #0f172a, #020617);
            color: var(--text);
            display: flex;
            justify-content: center;
            padding: 24px;
        }

        .app-shell {
            width: 100%;
            max-width: 1100px;
            background:
              radial-gradient(circle at 10% 20%, rgba(59,130,246,0.4) 0, transparent 40%),
              radial-gradient(circle at 90% 10%, rgba(56,189,248,0.35) 0, transparent 40%),
              radial-gradient(circle at 90% 90%, rgba(52,211,153,0.35) 0, transparent 40%),
              linear-gradient(145deg, #020617 0%, #020617 45%, #020617 100%);
            border-radius: 26px;
            border: 1px solid rgba(15,23,42,0.7);
            box-shadow:
              0 30px 80px rgba(15,23,42,0.82),
              0 0 0 1px rgba(148,163,184,0.15);
            position: relative;
            overflow: hidden;
        }

        .glow-orb {
            position: absolute;
            filter: blur(40px);
            opacity: 0.65;
            pointer-events: none;
            border-radius: 999px;
        }
        .glow-orb.orb-1 { width: 180px; height: 180px; background: rgba(59,130,246,0.4); top: -40px; left: 20%; }
        .glow-orb.orb-2 { width: 220px; height: 220px; background: rgba(56,189,248,0.4); top: -70px; right: 15%; }
        .glow-orb.orb-3 { width: 260px; height: 260px; background: rgba(52,211,153,0.4); bottom: -90px; right: 20%; }
        .glow-orb.orb-4 { width: 140px; height: 140px; background: rgba(239,68,68,0.4); bottom: -60px; left: 15%; }

        .glass-layer {
            position: absolute;
            inset: 0;
            background:
              radial-gradient(circle at top left, rgba(148,163,184,0.12), transparent 50%),
              radial-gradient(circle at bottom right, rgba(148,163,184,0.12), transparent 50%);
            mix-blend-mode: soft-light;
            pointer-events: none;
        }

        .page {
            position: relative;
            display: flex;
            z-index: 1;
        }

        .sidebar {
            width: 280px;
            padding: 26px 22px;
            border-right: 1px solid rgba(15,23,42,0.35);
            background: linear-gradient(180deg,
                        rgba(15,23,42,0.94),
                        rgba(15,23,42,0.97),
                        rgba(2,6,23,0.98));
            display: flex;
            flex-direction: column;
            gap: 22px;
        }
        :root[data-theme="light"] .sidebar {
            background: linear-gradient(180deg,
                        #f9fafb,
                        #f3f4f6);
            border-right-color: rgba(209,213,219,0.9);
        }
        @media (max-width: 960px) {
            .page {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
                border-right: none;
                border-bottom: 1px solid rgba(15,23,42,0.35);
            }
            :root[data-theme="light"] .sidebar {
                border-bottom-color: rgba(209,213,219,0.9);
            }
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.5);
            background: radial-gradient(circle at top left,
                         rgba(59,130,246,0.3),
                         transparent 60%);
            font-size: 11px;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: radial-gradient(circle, #22c55e 0, #166534 70%);
            box-shadow: 0 0 0 1px rgba(22,163,74,0.6),
                        0 0 16px rgba(74,222,128,0.9);
        }

        .app-title {
            margin-top: 2px;
            font-size: 22px;
            font-weight: 600;
            letter-spacing: -.03em;
        }
        .app-subtitle {
            font-size: 13px;
            color: var(--muted);
            margin-top: 6px;
            line-height: 1.5;
        }

        .meta-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 12px;
        }
        .chip {
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.5);
            font-size: 11px;
            color: var(--muted);
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(15,23,42,0.78);
        }
        :root[data-theme="light"] .chip {
            background: rgba(248,250,252,0.95);
        }
        .chip-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: rgba(59,130,246,0.9);
        }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 12px;
            border-top: 1px dashed rgba(148,163,184,0.4);
            font-size: 11px;
            color: var(--muted);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .footer-row {
            display: flex;
            justify-content: space-between;
            gap: 8px;
        }
        .footer-pill {
            padding: 3px 9px;
            border-radius: 999px;
            background: rgba(15,23,42,0.88);
            border: 1px solid rgba(148,163,184,0.5);
        }
        :root[data-theme="light"] .footer-pill {
            background: rgba(248,250,252,0.95);
        }

        .sidebar-kpi {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .kpi-pill {
            flex: 1;
            padding: 7px 10px;
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,0.5);
            background: linear-gradient(135deg,
                        rgba(15,23,42,0.96),
                        rgba(15,23,42,0.99));
            font-size: 11px;
        }
        :root[data-theme="light"] .kpi-pill {
            background: linear-gradient(135deg,
                        rgba(248,250,252,1),
                        rgba(241,245,249,1));
        }
        .kpi-label {
            color: var(--muted);
            font-size: 10px;
        }
        .kpi-value {
            margin-top: 4px;
            font-weight: 600;
            font-size: 13px;
        }

        .content {
            flex: 1;
            padding: 22px;
            backdrop-filter: blur(24px);
            background: radial-gradient(circle at top left,
                        rgba(15,23,42,0.80),
                        rgba(15,23,42,0.96));
        }
        :root[data-theme="light"] .content {
            background: radial-gradient(circle at top left,
                        rgba(248,250,252,0.96),
                        rgba(241,245,249,0.98));
        }

        .content-header {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: center;
            margin-bottom: 18px;
        }
        .content-title-block h1 {
            margin: 0;
            font-size: 18px;
            letter-spacing: -.02em;
        }
        .content-title-block p {
            margin: 4px 0 0;
            font-size: 12px;
            color: var(--muted);
        }

        .mode-toggle {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 8px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.6);
            background: radial-gradient(circle at top,
                         rgba(15,23,42,0.95),
                         rgba(15,23,42,0.99));
            font-size: 11px;
        }
        :root[data-theme="light"] .mode-toggle {
            background: rgba(248,250,252,0.97);
        }
        .mode-toggle button {
            border: none;
            background: none;
            color: var(--muted);
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 999px;
            cursor: pointer;
        }
        .mode-toggle button.active {
            background: rgba(15,23,42,0.9);
            color: #f9fafb;
        }
        :root[data-theme="light"] .mode-toggle button.active {
            background: #0f172a;
        }

        .main-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
            gap: 18px;
        }
        @media (max-width: 960px) {
            .main-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }

        .card {
            background: rgba(15,23,42,0.94);
            border-radius: 20px;
            border: 1px solid rgba(30,64,175,0.7);
            box-shadow:
              0 18px 45px rgba(15,23,42,0.85),
              0 0 0 1px rgba(30,64,175,0.4);
            position: relative;
            overflow: hidden;
        }
        :root[data-theme="light"] .card {
            background: linear-gradient(145deg,#ffffff,#f9fafb);
            border-color: rgba(209,213,219,0.9);
            box-shadow:
              0 10px 35px rgba(148,163,184,0.45),
              0 0 0 1px rgba(209,213,219,0.9);
        }
        .card-header {
            padding: 14px 16px 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(148,163,184,0.4);
        }
        .card-title-block h2 {
            margin: 0;
            font-size: 14px;
            letter-spacing: -.01em;
        }
        .card-title-block p {
            margin: 3px 0 0;
            font-size: 11px;
            color: var(--muted);
        }
        .card-body {
            padding: 14px 16px 16px;
        }

        .pill-bar {
            display: inline-flex;
            padding: 3px;
            border-radius: 999px;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(30,64,175,0.75);
            font-size: 11px;
            gap: 3px;
        }
        :root[data-theme="light"] .pill-bar {
            background: rgba(248,250,252,0.98);
            border-color: rgba(209,213,219,0.9);
        }
        .pill-option {
            padding: 4px 8px;
            border-radius: 999px;
            color: var(--muted);
        }
        .pill-option.active {
            background: linear-gradient(135deg,#3b82f6,#22c55e);
            color: #f9fafb;
        }

        .upload-wrapper {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
            gap: 18px;
        }
        @media (max-width: 780px) {
            .upload-wrapper {
                grid-template-columns: minmax(0, 1fr);
            }
        }

        .upload-zone {
            border-radius: 16px;
            border: 1px dashed rgba(148,163,184,0.6);
            background: radial-gradient(circle at top,
                         rgba(15,23,42,0.9),
                         rgba(15,23,42,0.98));
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        :root[data-theme="light"] .upload-zone {
            background: rgba(248,250,252,0.98);
            border-color: rgba(209,213,219,0.9);
        }
        .upload-zone-header {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .upload-icon {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: radial-gradient(circle,
                         rgba(59,130,246,0.5),
                         rgba(37,99,235,0.2));
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
        }
        .upload-text {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .upload-title {
            font-size: 13px;
            font-weight: 500;
        }
        .upload-hint {
            font-size: 11px;
            color: var(--muted);
        }

        .upload-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
        }
        .btn-primary {
            border: none;
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: .02em;
            background: linear-gradient(135deg,#3b82f6,#22c55e);
            color: #f9fafb;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow:
              0 10px 25px rgba(59,130,246,0.65),
              0 0 0 1px rgba(37,99,235,0.7);
        }
        :root[data-theme="light"] .btn-primary {
            box-shadow:
              0 10px 25px rgba(59,130,246,0.45),
              0 0 0 1px rgba(37,99,235,0.6);
        }
        .btn-primary:hover {
            filter: brightness(1.05);
            transform: translateY(-0.5px);
        }
        .btn-secondary {
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.6);
            padding: 6px 10px;
            background: rgba(15,23,42,0.96);
            color: var(--muted);
            font-size: 11px;
            cursor: pointer;
        }
        :root[data-theme="light"] .btn-secondary {
            background: rgba(248,250,252,0.98);
        }
        .file-input {
            display: none;
        }
        .file-name {
            margin-top: 6px;
            font-size: 11px;
            color: var(--muted);
        }

        .preview-card {
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.5);
            background: radial-gradient(circle at top,
                         rgba(15,23,42,0.92),
                         rgba(15,23,42,0.98));
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        :root[data-theme="light"] .preview-card {
            background: rgba(248,250,252,0.98);
            border-color: rgba(209,213,219,0.9);
        }
        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: var(--muted);
        }
        .preview-image-wrapper {
            border-radius: 12px;
            overflow: hidden;
            background: radial-gradient(circle, #020617, #000);
            border: 1px solid rgba(15,23,42,0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
        }
        .preview-image-wrapper img {
            max-width: 100%;
            max-height: 260px;
            object-fit: contain;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 12px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
        }
        .card-caption {
            font-size: 11px;
            color: var(--muted);
        }

        .upload-zone-compact {
            border-radius: 12px;
            border: 1px dashed rgba(148,163,184,0.6);
            padding: 9px 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 10px;
            background: rgba(15,23,42,0.98);
        }
        :root[data-theme="light"] .upload-zone-compact {
            background: rgba(248,250,252,0.98);
            border-color: rgba(209,213,219,0.9);
        }
        .upload-zone-compact span {
            font-size: 11px;
            color: var(--muted);
        }

        .main-prediction {
            border-radius: 14px;
            padding: 10px 12px;
            background: linear-gradient(135deg,
                         rgba(16,185,129,0.15),
                         rgba(16,185,129,0.08),
                         rgba(59,130,246,0.16));
            border: 1px solid rgba(34,197,94,0.6);
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
        }
        :root[data-theme="light"] .main-prediction {
            background: linear-gradient(135deg,
                         rgba(16,185,129,0.1),
                         rgba(59,130,246,0.12));
            border-color: rgba(34,197,94,0.7);
        }
        .main-prediction-label {
            font-size: 12px;
        }
        .main-prediction-value {
            font-size: 18px;
            font-weight: 600;
            letter-spacing: -.03em;
        }
        .main-prediction-tag {
            font-size: 11px;
            color: rgba(22,163,74,0.9);
        }

        .probability-list {
            margin-top: 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .prob-row {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        .prob-label {
            width: 80px;
        }
        .prob-bar {
            flex: 1;
            height: 8px;
            border-radius: 999px;
            background: rgba(15,23,42,0.85);
            overflow: hidden;
            border: 1px solid rgba(30,64,175,0.7);
        }
        :root[data-theme="light"] .prob-bar {
            background: rgba(229,231,235,0.95);
            border-color: rgba(209,213,219,0.9);
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg,
                         rgba(59,130,246,1),
                         rgba(52,211,153,1));
        }
        .prob-value {
            width: 46px;
            text-align: right;
            font-variant-numeric: tabular-nums;
            color: var(--muted);
            font-size: 11px;
        }

        .error-box {
            margin-top: 10px;
            border-radius: 12px;
            padding: 8px 10px;
            background: rgba(127,29,29,0.18);
            border: 1px solid rgba(248,113,113,0.7);
            font-size: 11px;
            color: #fecaca;
        }
        :root[data-theme="light"] .error-box {
            background: rgba(254,242,242,0.98);
            color: #991b1b;
            border-color: rgba(248,113,113,0.85);
        }

        .toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .toolbar-right {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: var(--muted);
        }
        .dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 0 1px rgba(34,197,94,0.7),
                        0 0 12px rgba(74,222,128,0.9);
        }
        .status-pill {
            padding: 4px 8px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.6);
            background: rgba(15,23,42,0.96);
        }
        :root[data-theme="light"] .status-pill {
            background: rgba(248,250,252,0.98);
        }

        .small-label {
            font-size: 10px;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .small-value {
            font-size: 12px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .stat-block {
            padding: 8px 9px;
            border-radius: 10px;
            border: 1px solid rgba(148,163,184,0.6);
            background: rgba(15,23,42,0.96);
            font-size: 11px;
        }
        :root[data-theme="light"] .stat-block {
            background: rgba(248,250,252,0.98);
        }
        .stat-label {
            color: var(--muted);
            font-size: 10px;
        }
        .stat-value {
            margin-top: 3px;
            font-weight: 500;
        }

        .credits {
            margin-top: 8px;
            font-size: 10px;
            color: var(--muted);
        }
        .credits a {
            color: var(--accent);
            text-decoration: none;
        }
        .credits a:hover {
            text-decoration: underline;
        }

        .theme-chip {
            padding: 3px 8px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.6);
            font-size: 10px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .theme-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: radial-gradient(circle, #fbbf24, #ea580c);
        }

        .preview-badge {
            padding: 3px 7px;
            border-radius: 999px;
            background: rgba(15,23,42,0.96);
            border: 1px solid rgba(148,163,184,0.6);
            font-size: 10px;
            color: var(--muted);
        }
        :root[data-theme="light"] .preview-badge {
            background: rgba(248,250,252,0.98);
        }
    </style>
</head>
<body>
<div class="app-shell">
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>
    <div class="glow-orb orb-3"></div>
    <div class="glow-orb orb-4"></div>
    <div class="glass-layer"></div>

    <div class="page">
        <aside class="sidebar">
            <div>
                <div class="badge">
                    <span class="badge-dot"></span>
                    <span>Real-time Inference</span>
                </div>
                <div class="app-title">Dog Emotion Classifier</div>
                <div class="app-subtitle">
                    Eğitimli CNN modelini doğrudan tarayıcıdan test et.
                    Bir köpek fotoğrafı yükle, duygusunu tahmin edelim.
                </div>

                <div class="meta-chips">
                    <div class="chip">
                        <span class="chip-dot"></span>
                        <span>PyTorch · Flask UI</span>
                    </div>
                    <div class="chip">
                        <span style="width:6px;height:6px;border-radius:999px;background:#22c55e;"></span>
                        <span>GPU aware</span>
                    </div>
                    <div class="chip">
                        <span style="width:6px;height:6px;border-radius:999px;background:#facc15;"></span>
                        <span>Multi-class emotions</span>
                    </div>
                </div>
            </div>

            <div class="sidebar-kpi">
                <div class="kpi-pill">
                    <div class="kpi-label">ACTIVE MODEL</div>
                    <div class="kpi-value">SimpleCNN · {{ all_probs|length if all_probs else '4' }} sınıf</div>
                </div>
                <div class="kpi-pill">
                    <div class="kpi-label">DEVICE</div>
                    <div class="kpi-value">{{ 'GPU / MPS' if 'cuda' in '{{ "" }}' or 'mps' in '{{ "" }}' else 'CPU' }}</div>
                </div>
            </div>

            <div class="sidebar-footer">
                <div class="footer-row">
                    <div class="footer-pill">
                        <span style="font-size:10px;">Latency ~50-100ms</span>
                    </div>
                    <div class="footer-pill">
                        <span style="font-size:10px;">Input: RGB image · {{ IMAGE_SIZE }}×{{ IMAGE_SIZE }}</span>
                    </div>
                </div>
                <div class="footer-row">
                    <span>Model eğitimi ayrı bir Python script'i ile yapıldı.</span>
                </div>
                <div class="footer-row">
                    <span>Bu arayüz, sadece inference (tahmin) için tasarlandı.</span>
                </div>
            </div>
        </aside>

        <main class="content">
            <header class="content-header">
                <div class="content-title-block">
                    <h1>Realtime inference workspace</h1>
                    <p>Resim yükle · Tahmin al · Olasılık dağılımını gör.</p>
                </div>
                <div class="mode-toggle">
                    <button type="button" data-theme-switch="dark" class="active">Dark</button>
                    <button type="button" data-theme-switch="light">Light</button>
                </div>
            </header>

            <section class="card">
                <div class="card-header">
                    <div class="card-title-block">
                        <h2>Upload & Preview</h2>
                        <p>Bir köpek fotoğrafı seç ve model üzerinde çalıştır.</p>
                    </div>
                    <div class="pill-bar">
                        <span class="pill-option active">Single image</span>
                        <span class="pill-option">Live demo</span>
                    </div>
                </div>
                <div class="card-body">
                    <form method="post" enctype="multipart/form-data">
                        <div class="upload-wrapper">
                            <div>
                                <div class="upload-zone">
                                    <div class="upload-zone-header">
                                        <div class="upload-icon">↑</div>
                                        <div class="upload-text">
                                            <div class="upload-title">
                                                Fotoğraf yükle ya da sürükleyip bırak
                                            </div>
                                            <div class="upload-hint">
                                                Desteklenen türler: JPG · PNG
                                            </div>
                                        </div>
                                    </div>
                                    <div class="upload-actions">
                                        <label class="btn-primary">
                                            <span>Dosya Seç</span>
                                            <input type="file" name="file" class="file-input"
                                                   accept="image/*"
                                                   onchange="onFileNameChange(this)">
                                        </label>
                                        <button type="submit" class="btn-secondary">
                                            Tahmin Et
                                        </button>
                                    </div>
                                    <div id="file-name" class="file-name">
                                        Henüz dosya seçilmedi.
                                    </div>
                                </div>
                            </div>

                            <div>
                                <div class="preview-card">
                                    <div class="preview-header">
                                        <span>Input preview</span>
                                        <span class="preview-badge">
                                            {{ IMAGE_SIZE }}×{{ IMAGE_SIZE }} · RGB
                                        </span>
                                    </div>
                                    <div class="preview-image-wrapper">
                                        {% if image_data %}
                                            <img src="data:image/png;base64,{{ image_data }}" alt="Uploaded Image">
                                        {% else %}
                                            <span style="font-size:11px; color:var(--muted);">
                                                Seçilen fotoğraf burada görünecek.
                                            </span>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {% if error %}
                        <div class="error-box">
                            {{ error }}
                        </div>
                        {% endif %}
                    </form>
                </div>
            </section>

            <section class="card" style="margin-top: 14px;">
                <div class="card-header">
                    <div class="card-title-block">
                        <h2>Prediction</h2>
                        <p>Tahmin edilen duygu ve olasılık dağılımı.</p>
                    </div>
                    <div class="toolbar">
                        <div class="toolbar-right">
                            <div class="dot"></div>
                            <span>{{ 'Ready' if not top_class else 'Prediction updated' }}</span>
                        </div>
                    </div>
                </div>
                <div class="card-body">
                    {% if top_class %}
                        <div class="main-prediction">
                            <div>
                                <div class="main-prediction-label">Tahmin edilen duygu</div>
                                <div class="main-prediction-value">
                                    {{ top_class|capitalize }}
                                </div>
                                <div class="main-prediction-tag">
                                    Model güveni: yaklaşık {{ (top_prob*100)|round(1) }}%
                                </div>
                            </div>
                            <div>
                                <div class="small-label">Model snapshot</div>
                                <div class="small-value">
                                    SimpleCNN · {{ all_probs|length }} sınıf
                                </div>
                            </div>
                        </div>

                        <div class="probability-list">
                            {% for label, prob in all_probs %}
                            <div class="prob-row">
                                <div class="prob-label">{{ label|capitalize }}</div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill" style="width: {{ (prob*100)|round(1) }}%;"></div>
                                </div>
                                <div class="prob-value">{{ (prob*100)|round(1) }}%</div>
                            </div>
                            {% endfor %}
                        </div>

                        <div class="stats-grid">
                            <div class="stat-block">
                                <div class="stat-label">Model türü</div>
                                <div class="stat-value">Convolutional Neural Network</div>
                            </div>
                            <div class="stat-block">
                                <div class="stat-label">Input shape</div>
                                <div class="stat-value">1 × 3 × {{ IMAGE_SIZE }} × {{ IMAGE_SIZE }}</div>
                            </div>
                        </div>
                    {% else %}
                        <p style="font-size:12px; color:var(--muted); margin:0;">
                            Henüz tahmin yok. Başlamak için bir köpek fotoğrafı yükle ve "Tahmin Et" butonuna tıkla.
                        </p>
                    {% endif %}

                    <div class="credits">
                        Not: Bu arayüz sadece inference amaçlıdır. Eğitim kodu,
                        data augmentations ve metric hesaplamaları ayrı bir Python
                        script'i içerisinde tanımlanmıştır.
                    </div>
                </div>
            </section>
        </main>
    </div>
</div>

<script>
function onFileNameChange(input) {
    const fileNameSpan = document.getElementById("file-name");
    if (input.files && input.files.length > 0) {
        fileNameSpan.textContent = "Seçilen dosya: " + input.files[0].name;
    } else {
        fileNameSpan.textContent = "Henüz dosya seçilmedi.";
    }
}

const rootEl = document.documentElement;
const buttons = document.querySelectorAll("[data-theme-switch]");
buttons.forEach(btn => {
    btn.addEventListener("click", () => {
        const theme = btn.getAttribute("data-theme-switch");
        rootEl.setAttribute("data-theme", theme);

        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
    });
});
</script>
</body>
</html>
"""


# Ana sayfa route'u.
# GET isteğinde sadece boş form gösterilir.
# POST isteğinde yüklenen resmi okuyup `predict_image` ile tahmin alır,
# sonucu ve resmi HTML template içine gömerek kullanıcıya gösterir.
@app.route("/", methods=["GET", "POST"])
def index():
    top_class = None
    top_prob = None
    all_probs = None
    image_data = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Dosya bulunamadı. Lütfen bir resim seç."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Lütfen geçerli bir dosya seç."
            else:
                try:
                    img_bytes = file.read()
                    pil_img = Image.open(BytesIO(img_bytes))

                    top_class, top_prob, all_probs = predict_image(pil_img)

                    buffered = BytesIO()
                    pil_img.convert("RGB").save(buffered, format="PNG")
                    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                except Exception as e:
                    error = f"Error while processing image: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        top_class=top_class,
        top_prob=top_prob,
        all_probs=all_probs,
        image_data=image_data,
        error=error,
    )
