import streamlit as st
import tempfile
import os
import cv2
import torch
import numpy as np
from model import CNN_LSTM
from dataset import VideoDataset
import torchvision.transforms as transforms
import time
import base64
from PIL import Image
import json

# Settings
SEQUENCE_LENGTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["normal", "suspicious"]

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.3


# Load Model
@st.cache_resource
def load_model():
    model = CNN_LSTM(hidden_dim=128, dropout=0.29, num_classes=2)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Function to extract 32 frames evenly
def extract_frames(video_path, seq_len=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len:
        st.error("Video too short. Please use a longer clip.")
        return None

    indices = np.linspace(0, total_frames - 1, seq_len).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToTensor()(frame)
        frame = transform(frame)
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()

    if len(frames) != seq_len:
        st.error("Failed to extract required frames.")
        return None

    return torch.stack(frames).unsqueeze(0)  # [1, 32, 3, 224, 224]

# Function to play local alarm sound
def play_alarm(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

# UI
st.title("🎥 Smart CCTV Suspicious Activity Detector")

input_type = st.radio("Choose input type:", ["Webcam (Live)", "Upload Video"])

if input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_tensor = extract_frames(tfile.name)

        if video_tensor is not None:
            with st.spinner("Analyzing video..."):
                with torch.no_grad():
                    output = model(video_tensor.to(DEVICE))
                    pred = torch.argmax(output, dim=1).item()
                    st.success(f"Prediction: {LABELS[pred]}")

                if pred == 1:
                    st.markdown("<h3 style='color:red'>⚠️ Suspicious Activity Detected!</h3>", unsafe_allow_html=True)
                    play_alarm("alarm.mp3")

else:
    st.subheader("Live Webcam Feed")
    if st.button("Capture 32 Frames & Analyze"):
        cap = cv2.VideoCapture(0)
        frames = []
        frame_count = 0
        frame_placeholder = st.empty()

        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            frame_tensor = transforms.ToTensor()(frame_rgb)
            frame_tensor = transform(frame_tensor)
            frames.append(frame_tensor)
            frame_count += 1
            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()

        if len(frames) == SEQUENCE_LENGTH:
            video_tensor = torch.stack(frames).unsqueeze(0)
            with st.spinner("Analyzing live frames..."):
                with torch.no_grad():
                    output = model(video_tensor.to(DEVICE))
                    pred = torch.argmax(output, dim=1).item()
                    st.success(f"Prediction: {LABELS[pred]}")

                if pred == 1:
                    st.markdown("<h3 style='color:red'>⚠️ Suspicious Activity Detected!</h3>", unsafe_allow_html=True)
                    play_alarm("alarm.mp3")

st.caption("Built with CNN + LSTM | PyTorch + Streamlit")
