import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# 1. Define the exact same AI architecture
class SARTargetNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SARTargetNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. Lee Despeckling Filter
def lee_filter(img, win_size=5):
    img = img.astype(np.float32)
    img_mean = cv2.blur(img, (win_size, win_size))
    img_sqr_mean = cv2.blur(img**2, (win_size, win_size))
    img_variance = np.maximum(0, img_sqr_mean - img_mean**2)
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-5)
    return np.clip(img_mean + img_weights * (img - img_mean), 0, 255).astype(np.uint8)

# 3. Load Trained Model
@st.cache_resource
def load_model():
    model = SARTargetNet(num_classes=4)
    model.load_state_dict(torch.load("sar_classifier.pth", map_location="cpu"))
    model.eval()
    return model

st.set_page_config(page_title="Tactical SAR ATR System", layout="wide")
st.title("🛰️ Tactical SAR Automatic Target Recognition (ATR)")
st.markdown("**Human-in-the-Loop Decision Support System** for All-Weather Radar Reconnaissance.")

class_names = ['Main Battle Tank', 'Armored Carrier', 'Missile Launcher', 'Supply Truck']

uploaded_file = st.file_uploader("Upload a raw X-Band SAR Radar Scan (.png / .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    raw_img = np.array(image)
    clean_img = lee_filter(raw_img)
    
    col1, col2 = st.columns(2)
    col1.image(raw_img, caption="1. Raw Noisy SAR Input", use_container_width=True)
    col2.image(clean_img, caption="2. Lee-Filtered Despeckled Output", use_container_width=True)
    
    # Run AI Inference
    model = load_model()
    tensor_img = torch.tensor(clean_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        
    st.success(f"### 🎯 Detected Target: **{class_names[pred_idx]}** (Confidence: **{conf.item()*100:.2f}%**)")
    st.warning("⚠️ **Human-in-the-Loop Protocol:** AI classification requires operator verification before tactical action.")
