# Emotion Detector

Real-time facial emotion detection using OpenCV and a pretrained mini-XCEPTION model trained on the FER-2013 dataset.

## Demo

Detects faces from your webcam and classifies emotions live with a confidence score and bar chart overlay.

**Supported emotions:** angry, disgust, fear, happy, sad, surprise, neutral

## How It Works

```
Camera frame → Haar Cascade (face detection) → mini-XCEPTION model → emotion + confidence
```

- **Face detection** — OpenCV Haar Cascade
- **Emotion model** — mini-XCEPTION CNN pretrained on FER-2013 (~66% accuracy)
- **Display** — colored bounding box + label + live probability bar chart

## Project Structure

```
emotion-detector/
├── model.py              # mini-XCEPTION architecture
├── emotion_detector.py   # main real-time detection script
├── download_model.py     # downloads pretrained weights
└── requirements.txt
```

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download pretrained weights (~26MB)**
```bash
python download_model.py
```

**3. Run**
```bash
python emotion_detector.py
```

Use a different camera index if needed:
```bash
python emotion_detector.py 1
```

Press **Q** to quit.

## Model

Mini-XCEPTION is a lightweight CNN using depthwise separable convolutions and residual connections, making it fast enough for real-time inference.

- Input: 64×64 grayscale face crop
- Output: 7 softmax probabilities (one per emotion)
- Architecture reference: [Arriaga et al. (2017)](https://arxiv.org/abs/1710.07557)
- Weights source: [oarriaga/face_classification](https://github.com/oarriaga/face_classification)

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.7+
