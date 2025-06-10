# 💤 Alertic - Drowsiness Detection System

The **ALERTIC** is an AI-powered safety solution designed to monitor and detect signs of driver fatigue or drowsiness using computer vision and deep learning techniques. Built with OpenCV, TensorFlow, and Keras, this project classifies facial features into alert, drowsy, or sleepy states based on real-time video analysis.

---

## 📌 Key Features

- 👁️ Real-time eye and facial expression tracking
- 🤖 Deep learning model (CNN) for high-accuracy drowsiness classification
- 🎥 Video stream integration using OpenCV
- 🔔 Visual alert system for detected drowsiness
- 💻 Cloud-based training via Google Colab

---

## 🛠️ Technologies Used

- **Language:** Python
- **Libraries:** OpenCV, TensorFlow, Keras, NumPy, Matplotlib, Seaborn
- **Model:** Custom Convolutional Neural Network (CNN)
- **Environment:** Google Colab

---

## 🧠 Dataset

- **Classes:** Sleepy (1200 images), Drowsy (1000 images), Awake/Active (1500 images)
- **Source:** Public human facial datasets
- **Augmentation:** Flipping, rotation, brightness, cropping

---

## 🧱 Model Architecture

- **Input Layer:** 224x224 grayscale images
- **Conv Layers:** Two convolutional blocks with ReLU and max-pooling
- **Normalization:** BatchNormalization for training stability
- **Dense Layers:** Fully connected layers with Dropout
- **Output Layer:** Softmax activation with 3 output classes

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/drowsiness-detection.git
   cd drowsiness-detection
2. **Install Dependencies**
   ```bash 
   pip install -r requirements.txt
3. **Run the Model on Test Images**
   ```bash
   python test_model.py --img_path "path/to/image.jpg"
4. **Run Real-time Detection**
   ```bash 
   python detect_drowsiness_live.py

## 💡 Make sure your webcam is accessible and YOLOv8/Keras model weights are in the correct folder.

---

## 📊 Performance Metrics
Metric	Training Set	Validation Set
Accuracy	96%	91%
Precision	94%	89%
Recall	95%	90%
F1-Score	94%	89%

---

## 🧪 Testing & Evaluation
Evaluation using a confusion matrix and classification report

Real-world testing in various lighting conditions

Threshold tuning for reducing false positives

---

## 🧩 Challenges Faced
Variability in lighting and facial features

Overfitting — mitigated using data augmentation and dropout

Difficulty distinguishing small eyes vs closed eyes in certain frames

---

## 🔮 Future Scope
Integration with IoT devices for in-car alert systems

Edge device optimization (Raspberry Pi, Jetson Nano)

Incorporating video temporal patterns for improved detection

Multimodal sensor integration (e.g., yawning detection, pulse)

---

## 🙌 Acknowledgments
This project was created as part of the 5th semester Practical Training for the B.Tech (CSE-AI&ML) program at Dronacharya College of Engineering, Farukhnagar, Haryana.

---

## 🔗 References
OpenCV Documentation

TensorFlow

Keras

Google Colab

Dataset on Kaggle
