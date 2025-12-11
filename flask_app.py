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
print("Classes:", class_names)


# ==============================
# 5. Model load
# ==============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model dosyası bulunamadı: {MODEL_PATH}\n"
        f"Önce train_dog_emotion.py çalıştırıp best_model.pt üretmelisin."
    )

model = SimpleCNN(num_classes=num_classes)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ==============================
# 6. Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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
            --bg: #050816;
            --bg-soft: #0b1020;
            --card: #0f172a;
            --card-soft: #111827;
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
            --text: #020617;
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
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(148,163,184,0.25), transparent 55%),
                radial-gradient(circle at bottom right, rgba(37,99,235,0.18), transparent 55%),
                var(--bg);
            transition: background 0.35s ease, color 0.25s ease;
        }
        .page {
            display: flex;
            min-height: 100vh;
            backdrop-filter: blur(20px);
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
        .brand {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .brand-main {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-logo {
            width: 36px;
            height: 36px;
            border-radius: 14px;
            background: radial-gradient(circle at 30% 0%, #38bdf8, transparent 55%),
                        radial-gradient(circle at 70% 100%, #22c55e, transparent 55%),
                        #020617;
            border: 1px solid rgba(148,163,184,0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        :root[data-theme="light"] .brand-logo {
            border-color: rgba(148,163,184,0.6);
        }
        .brand-title {
            font-size: 15px;
            font-weight: 600;
        }
        .brand-sub {
            font-size: 11px;
            color: var(--muted);
        }

        .theme-toggle {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            border-radius: 999px;
            padding: 4px 10px;
            border: 1px solid rgba(148,163,184,0.5);
            background: rgba(15,23,42,0.85);
            cursor: pointer;
        }
        :root[data-theme="light"] .theme-toggle {
            background: rgba(255,255,255,0.85);
            border-color: rgba(209,213,219,0.9);
        }
        .theme-toggle-knob {
            width: 16px;
            height: 16px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #fde047, #facc15);
            box-shadow: 0 0 0 1px rgba(15,23,42,0.3);
        }
        :root[data-theme="dark"] .theme-toggle-knob {
            background: radial-gradient(circle at 30% 30%, #38bdf8, #0ea5e9);
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 6px;
        }
        .info-block {
            background: rgba(15,23,42,0.93);
            border-radius: 16px;
            padding: 12px 12px;
            border: 1px solid rgba(31,41,55,0.95);
        }
        :root[data-theme="light"] .info-block {
            background: rgba(255,255,255,0.9);
            border-color: rgba(209,213,219,0.9);
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 6px;
            color: var(--muted);
        }
        .info-row strong {
            color: var(--text);
            font-weight: 500;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .chip {
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(55,65,81,0.85);
            background: rgba(15,23,42,0.96);
            color: var(--muted);
        }
        :root[data-theme="light"] .chip {
            background: #f9fafb;
            border-color: #d1d5db;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 11px;
            background: rgba(15,23,42,0.9);
            border: 1px solid var(--border);
            color: var(--muted);
        }
        :root[data-theme="light"] .pill {
            background: #ffffff;
        }
        .pill-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .main {
            flex: 1;
            padding: 24px 26px 28px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .main-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
        }
        .main-title {
            font-size: 22px;
            font-weight: 620;
        }
        .main-subtitle {
            margin-top: 4px;
            font-size: 13px;
            color: var(--muted);
        }
        .main-meta {
            font-size: 11px;
            color: var(--muted);
            text-align: right;
        }
        .main-meta span {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(15,23,42,0.95);
            border: 1px solid var(--border);
        }
        :root[data-theme="light"] .main-meta span {
            background: #ffffff;
            border-color: #d1d5db;
        }
        .main-meta span strong {
            color: var(--accent-strong);
        }

        .content-grid {
            display: grid;
            gap: 16px;
            grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
        }
        @media (max-width: 1040px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: radial-gradient(circle at top left,
                        rgba(15,23,42,0.9),
                        rgba(15,23,42,0.97));
            border-radius: 18px;
            padding: 16px 16px 18px;
            border: 1px solid rgba(31,41,55,0.95);
            box-shadow: 0 20px 40px rgba(0,0,0,0.45);
            transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.25s ease;
        }
        :root[data-theme="light"] .card {
            background: #ffffff;
            border-color: #d1d5db;
            box-shadow: 0 18px 35px rgba(15,23,42,0.08);
        }
        .card:hover {
            transform: translateY(-1px);
            box-shadow: 0 25px 60px rgba(0,0,0,0.5);
        }
        :root[data-theme="light"] .card:hover {
            box-shadow: 0 24px 60px rgba(15,23,42,0.12);
        }
        .card-header {
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

        .upload-zone {
            margin-top: 4px;
            border-radius: 14px;
            border: 1px dashed rgba(148,163,184,0.7);
            background: linear-gradient(135deg,
                        rgba(15,23,42,0.96),
                        rgba(15,23,42,0.99));
            padding: 18px 16px;
            text-align: center;
        }
        :root[data-theme="light"] .upload-zone {
            background: linear-gradient(135deg, #f9fafb, #ffffff);
            border-color: #cbd5f5;
        }
        .upload-zone p {
            margin: 0 0 10px;
            font-size: 13px;
            color: var(--muted);
        }
        .file-input {
            margin: 8px 0 16px;
        }
        input[type="file"] {
            font-size: 12px;
            color: var(--muted);
        }
        .primary-btn {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 9px 22px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 12px 30px rgba(37,99,235,0.45);
        }
        :root[data-theme="light"] .primary-btn {
            box-shadow: 0 12px 28px rgba(37,99,235,0.36);
        }
        .primary-btn:hover {
            filter: brightness(1.06);
        }
        .primary-btn span {
            font-size: 14px;
        }
        .error {
            margin-top: 10px;
            font-size: 12px;
            color: var(--danger);
        }

        .preview-wrapper {
            margin-top: 16px;
            border-radius: 14px;
            background: radial-gradient(circle at top,
                        rgba(15,23,42,1),
                        rgba(2,6,23,1));
            border: 1px solid rgba(31,41,55,0.95);
            overflow: hidden;
        }
        :root[data-theme="light"] .preview-wrapper {
            background: #f9fafb;
            border-color: #d1d5db;
        }
        .preview-header {
            padding: 8px 12px;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            background: rgba(15,23,42,0.98);
            border-bottom: 1px solid rgba(31,41,55,0.95);
        }
        :root[data-theme="light"] .preview-header {
            background: #f3f4f6;
            border-bottom-color: #e5e7eb;
        }
        .preview-body {
            padding: 12px;
            text-align: center;
        }
        .preview-img {
            max-width: 100%;
            max-height: 260px;
            border-radius: 10px;
            box-shadow: 0 16px 35px rgba(0,0,0,0.6);
        }
        :root[data-theme="light"] .preview-img {
            box-shadow: 0 14px 30px rgba(148,163,184,0.55);
        }

        .summary-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .pred-block {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .pred-emoji {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(55,65,81,0.95);
        }
        :root[data-theme="light"] .pred-emoji {
            background: #ffffff;
            border-color: #d1d5db;
        }
        .pred-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--muted);
        }
        .pred-emotion {
            font-size: 20px;
            font-weight: 700;
            color: var(--accent-strong);
        }
        .confidence {
            font-size: 11px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.6);
            background: rgba(22,163,74,0.14);
            color: #bbf7d0;
        }
        :root[data-theme="light"] .confidence {
            background: rgba(22,163,74,0.12);
            color: #166534;
        }
        .summary-sub {
            margin-top: 8px;
            font-size: 12px;
            color: var(--muted);
        }

        .probs-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 12px;
        }
        .probs-table th,
        .probs-table td {
            padding: 6px 4px;
        }
        .probs-table thead {
            border-bottom: 1px solid rgba(31,41,55,0.95);
            color: var(--muted);
        }
        :root[data-theme="light"] .probs-table thead {
            border-bottom-color: #e5e7eb;
        }
        .label-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border-radius: 999px;
            background: rgba(15,23,42,0.98);
            border: 1px solid rgba(55,65,81,0.9);
        }
        :root[data-theme="light"] .label-badge {
            background: #f3f4f6;
            border-color: #d1d5db;
        }
        .label-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
        }
        .dot-angry { background: #f97373; }
        .dot-happy { background: #fbbf24; }
        .dot-relaxed { background: #38bdf8; }
        .dot-sad { background: #6366f1; }

        .bar-wrapper {
            width: 100%;
            background: rgba(17,24,39,0.95);
            border-radius: 999px;
            overflow: hidden;
            height: 8px;
        }
        :root[data-theme="light"] .bar-wrapper {
            background: #e5e7eb;
        }
        .bar-inner {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #3b82f6, #22c55e);
            transition: width 0.35s ease;
        }
        .prob-cell {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            font-size: 11px;
            color: var(--muted);
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
        }
        .legend-angry { background: #f97373; }
        .legend-happy { background: #fbbf24; }
        .legend-relaxed { background: #38bdf8; }
        .legend-sad { background: #6366f1; }

        .empty-state {
            margin-top: 6px;
            font-size: 12px;
            color: var(--muted);
        }
    </style>
</head>
<body>
<div class="page">
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="brand">
            <div class="brand-main">
                <div class="brand-logo">🐶</div>
                <div>
                    <div class="brand-title">Dog Emotion Classifier</div>
                    <div class="brand-sub">Deep Learning Inference UI</div>
                </div>
            </div>
            <button type="button" class="theme-toggle" id="themeToggle">
                <div class="theme-toggle-knob"></div>
                <span id="themeToggleLabel">Dark</span>
            </button>
        </div>

        <div>
            <div class="section-title">Model</div>
            <div class="info-block">
                <div class="info-row">
                    <span>Architecture</span>
                    <strong>SimpleCNN</strong>
                </div>
                <div class="info-row">
                    <span>Backend</span>
                    <strong>PyTorch</strong>
                </div>
                <div class="info-row">
                    <span>Classes</span>
                    <strong>4</strong>
                </div>
                <div class="chip-row">
                    <div class="chip">Convolutional layers</div>
                    <div class="chip">BatchNorm</div>
                    <div class="chip">Dropout</div>
                    <div class="chip">Softmax output</div>
                </div>
            </div>
        </div>

        <div>
            <div class="section-title">Emotion labels</div>
            <div class="chip-row">
                <div class="chip">
                    <span class="legend-dot legend-angry"></span> angry
                </div>
                <div class="chip">
                    <span class="legend-dot legend-happy"></span> happy
                </div>
                <div class="chip">
                    <span class="legend-dot legend-relaxed"></span> relaxed
                </div>
                <div class="chip">
                    <span class="legend-dot legend-sad"></span> sad
                </div>
            </div>
        </div>

        <div>
            <div class="section-title">Runtime</div>
            <span class="pill">
                <span class="pill-dot"></span>
                Inference on local device
            </span>
        </div>
    </aside>

    <!-- Main content -->
    <main class="main">
        <div class="main-header">
            <div>
                <div class="main-title">Inference Console</div>
                <div class="main-subtitle">
                    Upload a dog image, run inference, and inspect the full probability distribution over emotion labels.
                </div>
            </div>
            <div class="main-meta">
                <span>
                    Output space: <strong>angry · happy · relaxed · sad</strong>
                </span>
            </div>
        </div>

        <div class="content-grid">
            <!-- LEFT: Upload & preview -->
            <section class="card">
                <div class="card-header">
                    <div class="card-title">Image Input</div>
                    <div class="card-caption">JPG / PNG · RGB</div>
                </div>

                <form method="post" enctype="multipart/form-data">
                    <div class="upload-zone">
                        <p>Select a dog image from your local machine and start inference.</p>
                        <div class="file-input">
                            <input type="file" name="file" accept="image/*">
                        </div>
                        <button type="submit" class="primary-btn">
                            <span>▶</span> Run Inference
                        </button>
                        {% if error %}
                            <div class="error">{{ error }}</div>
                        {% endif %}
                    </div>
                </form>

                {% if image_data %}
                    <div class="preview-wrapper">
                        <div class="preview-header">Input preview</div>
                        <div class="preview-body">
                            <img src="data:image/png;base64,{{ image_data }}" class="preview-img">
                        </div>
                    </div>
                {% endif %}
            </section>

            <!-- RIGHT: Results -->
            <section class="card">
                <div class="card-header">
                    <div class="card-title">Prediction</div>
                    <div class="card-caption">Probability distribution across emotion classes</div>
                </div>

                {% if top_class %}
                    {% set top_emoji = "🐶" %}
                    {% if "angry" in top_class.lower() %}
                        {% set top_emoji = "😡" %}
                    {% elif "happy" in top_class.lower() %}
                        {% set top_emoji = "😄" %}
                    {% elif "relaxed" in top_class.lower() %}
                        {% set top_emoji = "😌" %}
                    {% elif "sad" in top_class.lower() %}
                        {% set top_emoji = "😢" %}
                    {% endif %}

                    <div class="summary-row">
                        <div class="pred-block">
                            <div class="pred-emoji">{{ top_emoji }}</div>
                            <div>
                                <div class="pred-label">Detected emotion</div>
                                <div class="pred-emotion">{{ top_class }}</div>
                            </div>
                        </div>
                        <div class="confidence">
                            Confidence: {{ (top_prob * 100) | round(1) }}%
                        </div>
                    </div>
                    <div class="summary-sub">
                        The table below summarizes the model's confidence for each emotion label.
                    </div>

                    <table class="probs-table">
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th style="width: 55%;">Probability</th>
                                <th style="text-align: right;">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for label, prob in all_probs %}
                            <tr>
                                <td>
                                    <span class="label-badge">
                                        {% set lname = label.lower() %}
                                        <span class="label-dot
                                            {% if 'angry' in lname %}dot-angry{% endif %}
                                            {% if 'happy' in lname %}dot-happy{% endif %}
                                            {% if 'relaxed' in lname %}dot-relaxed{% endif %}
                                            {% if 'sad' in lname %}dot-sad{% endif %}
                                        "></span>
                                        {{ label }}
                                    </span>
                                </td>
                                <td>
                                    <div class="bar-wrapper">
                                        <div class="bar-inner" style="width: {{ prob * 100 }}%;"></div>
                                    </div>
                                </td>
                                <td class="prob-cell">
                                    {{ (prob * 100) | round(1) }}%
                                </td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>

                    <div class="legend">
                        <span class="legend-item">
                            <span class="legend-dot legend-angry"></span> angry
                        </span>
                        <span class="legend-item">
                            <span class="legend-dot legend-happy"></span> happy
                        </span>
                        <span class="legend-item">
                            <span class="legend-dot legend-relaxed"></span> relaxed
                        </span>
                        <span class="legend-item">
                            <span class="legend-dot legend-sad"></span> sad
                        </span>
                    </div>
                {% else %}
                    <div class="empty-state">
                        No predictions yet. Use the form on the left to upload an image and run inference.
                    </div>
                {% endif %}
            </section>
        </div>
    </main>
</div>

<script>
(function() {
    const root = document.documentElement;
    const toggle = document.getElementById("themeToggle");
    const label = document.getElementById("themeToggleLabel");

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        if (theme === "light") {
            label.textContent = "Light";
        } else {
            label.textContent = "Dark";
        }
        try {
            window.localStorage.setItem("dog-emotion-theme", theme);
        } catch (e) {}
    }

    let savedTheme = null;
    try {
        savedTheme = window.localStorage.getItem("dog-emotion-theme");
    } catch (e) {}

    if (savedTheme === "light" || savedTheme === "dark") {
        applyTheme(savedTheme);
    } else {
        applyTheme("dark");
    }

    toggle.addEventListener("click", function() {
        const current = root.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
    });
})();
</script>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    top_class = None
    top_prob = None
    all_probs = None
    image_data = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file received."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Please select an image file."
            else:
                try:
                    image_bytes = file.read()
                    pil_img = Image.open(BytesIO(image_bytes))
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)




