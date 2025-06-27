🧐 Smart CCTV Surveillance System 🔍📵

A real-time AI-powered surveillance solution that detects **suspicious human activity** using deep learning. This system enhances traditional CCTV infrastructure by enabling **automated video analysis**, providing alerts upon detecting unusual behavior — improving safety, reducing monitoring costs, and supporting proactive response.

---

 🚀 Features

* 🎯 **Real-time Suspicious Activity Detection**
* 🧠 Powered by **CNN + LSTM neural networks**
* 📆 Local video upload + 📸 Live webcam support
* 🔊 Instant alerts via built-in **audio alarms**
* 🔻 User-friendly interface built with **Streamlit**
* 🖼️ Efficient frame extraction and sequence modeling

---

🛠️ Tech Stack

**Programming & Libraries:**

* Python
* PyTorch (Deep Learning)
* TorchVision (Pretrained ResNet)
* OpenCV (Video & webcam processing)
* Streamlit (Web UI)
* NumPy, PIL, tempfile, base64

**Neural Network Models:**

* 🧠 **CNN** (Convolutional Neural Network) — using **ResNet-18** for feature extraction
* 🖁️ **LSTM** (Long Short-Term Memory) — for learning temporal patterns in frame sequences

**Hardware Acceleration:**

* ⚡ **CUDA (GPU-accelerated training)** via PyTorch backend for faster training and inference

---

🧪 Training Configuration

* **Model**: ResNet18 + 2-layer LSTM
* **Hidden Size**: 128
* **Dropout**: 0.29
* **Sequence Length**: 24 frames
* **Learning Rate**: 9.6e-5
* **Batch Size**: 4
* **Epochs**: 15
* **Optimizer**: Adam
* **Loss**: CrossEntropyLoss

---

🎓 Use Cases

* Smart Campus Monitoring
* Industrial Safety Surveillance
* Night Patrol Automation
* Home Security Systems

---

ScreenShot 1 : 
![Screenshot 2025-06-27 223118](https://github.com/user-attachments/assets/cf20bc6e-f5f1-4aab-8f6c-416ee49de64c)

ScreenShot 2 : 
![Screenshot 2025-06-27 223126](https://github.com/user-attachments/assets/005cfda9-80d0-48df-b7ca-9593bb042992)


👨‍💼 Contributed By

> Hafeez Shaik
