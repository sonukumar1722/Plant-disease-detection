# 🌿 Plant Disease Recognition & Weather Forecast System

![GitHub last commit](https://img.shields.io/github/last-commit/sonukumar1722/Plant-disease-detection)
![GitHub repo size](https://img.shields.io/github/repo-size/sonukumar1722/Plant-disease-detection)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)

An intelligent web application that combines **deep learning** and **real-time weather data** to help farmers identify plant diseases and access critical agricultural information.

### 🚀 [**Try the Live App**](https://plant-disease-detection-0.streamlit.app/)

---

## 📑 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [Dataset Information](#-dataset-information)

---

## ✨ Features

### 🔬 Plant Disease Detection
- **Deep Learning Model**: TensorFlow/Keras CNN trained on ~87K images
- **38 Disease Classes**: Covers 14 different plant species
- **Multiple Input Methods**: Upload images or use live camera capture
- **Instant Analysis**: Real-time disease prediction with confidence levels
- **Comprehensive Treatment Information**: Disease descriptions, treatment recommendations, and prevention strategies

### 🌤️ Weather Forecast
- **5-Day Forecast**: Integrated OpenWeatherMap API
- **Agricultural Metrics**: Temperature, humidity, wind speed, rainfall predictions
- **Smart Alerts**: Warnings for high humidity, strong winds, and significant rainfall
- **Location-Based**: Search any location worldwide

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend/UI** | Streamlit 1.52.0+ |
| **Backend** | Python 3.8+ |
| **Machine Learning** | TensorFlow 2.20.0+, Keras |
| **Data Processing** | NumPy, Pandas, PIL (Pillow), OpenCV |
| **API Integration** | Requests, OpenWeatherMap API |
| **Security** | bcrypt, python-dotenv |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenWeatherMap API key ([Get free API key](https://openweathermap.org/api))

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sonukumar1722/Plant-disease-detection.git
   cd Plant-disease-detection
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENWEATHERMAP_API_KEY=your_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run main.py
   ```

6. **Access the Application**
   
   Open your browser at: `http://localhost:8501`

---

## 🎯 Usage Guide

### Disease Prediction

1. Navigate to **Disease Prediction** page from the sidebar
2. Choose input method:
   - **Upload Image**: Select a plant image (JPG, JPEG, PNG)
   - **Use Camera**: Capture a live image
3. Click **"🔍 Predict Disease"** button
4. Review results with treatment and prevention recommendations

### Weather Forecast

1. Navigate to **Weather Forecast** page from the sidebar
2. Enter your city name
3. Click **"Get Weather Forecast"**
4. Review 5-day forecast with agricultural metrics and smart alerts

---

## 🌱 Supported Plants & Diseases

### 38 Classes Across 14 Plant Species

| Plant | Diseases/Conditions |
|-------|---------------------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 📊 Dataset Information

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total Images**: ~87,000 RGB images
- **Training Set**: 70,295 images (80%)
- **Validation Set**: 17,572 images (20%)
- **Image Size**: 128x128 pixels
- **Classes**: 38 categories (diseases + healthy)

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - For comprehensive disease image data
- **OpenWeatherMap** - For weather API access
- **TensorFlow/Keras & Streamlit** - For excellent frameworks

---

## 📧 Contact

**Project Maintainer**: [sonukumar1722](https://github.com/sonukumar1722)

**Repository**: [Plant-disease-detection](https://github.com/sonukumar1722/Plant-disease-detection)

**Live App**: [https://plant-disease-detection-0.streamlit.app/](https://plant-disease-detection-0.streamlit.app/)

For questions or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for farmers and agricultural enthusiasts**

*Protecting crops, one prediction at a time* 🌾

![Repository Stats](https://img.shields.io/github/stars/sonukumar1722/Plant-disease-detection?style=social)
![Forks](https://img.shields.io/github/forks/sonukumar1722/Plant-disease-detection?style=social)

</div>
