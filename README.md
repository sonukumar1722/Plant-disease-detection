# 🌱 Plant Disease Recognition & Weather Forecast System, [launch Application](https://plant-disease-by-max.streamlit.app/)

![GitHub last commit](https://img.shields.io/github/last-commit/sonukumar1722/plant-disease-recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/sonukumar1722/plant-disease-recognition)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

An interactive web application that leverages **deep learning** and **real-time weather APIs** to help farmers, researchers, and agricultural enthusiasts identify plant diseases and access weather forecasts for informed crop management decisions.

---

## 📑 Table of Contents

* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [Dataset](#-dataset)
* [Installation & Setup](#-installation--setup)
* [Screenshots](#-screenshots)
* [Demo](#-demo)
* [Future Improvements](#-future-improvements)
* [Contributing](#-contributing)
* [License](#-license)

---

## 📌 Features

* **Plant Disease Detection**

  * TensorFlow CNN model trained on \~87K images from the Keras **PlantVillage** dataset.
  * Supports **image uploads** and **live camera capture** for predictions.
  * Delivers instant results with high classification accuracy.

* **Weather Forecast**

  * Integrates **OpenWeatherMap API** for **7-day weather forecasts**.
  * Displays temperature, humidity, wind speed, and conditions in a clean UI.

* **User-Friendly Interface**

  * Built with **Streamlit** for a responsive and intuitive design.
  * Sidebar navigation for quick access to all pages.

---

## 🛠 Tech Stack

* **Frontend / UI:** Streamlit
* **Backend / Logic:** Python, TensorFlow, NumPy, Pandas, Requests
* **Machine Learning:** CNN model (`trained_plant_disease_model.keras`)
* **APIs:** OpenWeatherMap API
* **Camera Integration:** `streamlit_back_camera_input`
* **Hosting:** Streamlit Sharing / Local deployment

---

## 📂 Dataset

* **Source:** [Keras / TensorFlow Datasets – PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village)
* **Description:** \~87K RGB images of healthy and diseased crop leaves.
* **Classes:** 38 categories including apple, corn, grape, potato, soybean, tomato, and more.
* **Split:** 80% training, 20% validation, with an additional test set for evaluation.

---

## 🚀 Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sonukumar1722/plant-disease-recognition.git
   cd plant-disease-recognition
   ```

2. **Create & Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add the Trained Model**

   * Place `trained_plant_disease_model.keras` in the project root directory.

5. **Set API Key**

   * Replace `API_KEY` in `app.py` with your OpenWeatherMap API key.

6. **Run the App**

   ```bash
   streamlit run app.py
   ```


## 📈 Future Improvements

* Multi-language support for farmers.
* Disease-specific treatment recommendations.
* Offline model predictions using **TensorFlow Lite**.

---


