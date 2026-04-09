# 🧠 Brain Tumor Detection AI

<p align="center">
  <img src="https://img.shields.io/badge/AI-Medical%20Imaging-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge">
  <img src="https://img.shields.io/badge/Frontend-JavaScript-orange?style=for-the-badge">
</p>

<p align="center">
  <b>AI-powered system to detect brain tumors from MRI scans with explainable predictions</b>
</p>

---

## 🚀 Overview

This project uses **Deep Learning + Computer Vision** to detect brain tumors from MRI images.
It provides **real-time predictions**, **Grad-CAM visual explanations**, and **downloadable medical reports**.

---

## ✨ Features

✅ Brain tumor detection using CNN
✅ Transfer Learning with MobileNetV2
✅ Grad-CAM visualization (Explainable AI)
✅ Interactive web UI (Flask + JS)
✅ PDF medical report generation
✅ Clinical insights & probability analysis

---

## 🖥️ Demo

> Upload MRI → Get prediction → View heatmap → Download report

📌 *(Add your screenshots here)*

---

## 🧠 Tech Stack

| Layer         | Technology            |
| ------------- | --------------------- |
| Model         | TensorFlow / Keras    |
| CV            | OpenCV                |
| Backend       | Flask                 |
| Frontend      | HTML, CSS, JavaScript |
| Visualization | Matplotlib            |
| Reports       | ReportLab             |

---

## 📂 Project Structure

```
Brain-Tumor-Detection-AI/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── data_loader.py
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── script.js
│   └── style.css
│
├── app.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Detection-AI.git
cd Brain-Tumor-Detection-AI
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

Then open 👉 http://localhost:5000

---

## 🧪 Model Details

* Architecture: **MobileNetV2 (Transfer Learning)**
* Input Size: **224x224**
* Output: **Binary Classification (Tumor / No Tumor)**
* Activation: **Sigmoid**

---

## 📊 Explainable AI

Grad-CAM is used to highlight **important regions in MRI scans** that influenced the prediction.

---

## 📄 Report Generation

* Patient details
* Prediction results
* Confidence metrics
* Doctor notes
* Downloadable PDF

---

## ⚠️ Disclaimer

This system is **AI-assisted** and should not be used as a sole diagnostic tool.
Always consult a qualified medical professional.

---

## 📌 Future Improvements

* Multi-class tumor classification
* Real-time deployment (cloud)
* Improved dataset & accuracy
* Doctor dashboard

---

## 👨‍💻 Author

**Rupas J**
AI & Software Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
