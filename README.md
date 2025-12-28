# 🌿 Plant Disease Recognition & Weather Forecast System

![GitHub last commit](https://img.shields.io/github/last-commit/sonukumar1722/Plant-disease-detection)
![GitHub repo size](https://img.shields.io/github/repo-size/sonukumar1722/Plant-disease-detection)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)

An intelligent web application that combines **deep learning** and **real-time weather data** to help farmers, agricultural researchers, and plant enthusiasts identify plant diseases and access critical weather information for informed agricultural decision-making.

---

## 📑 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Dependencies](#-dependencies)
- [File Descriptions](#-file-descriptions)
- [Dataset Information](#-dataset-information)
- [Usage Guide](#-usage-guide)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [How It Works](#-how-it-works)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🔬 Plant Disease Detection
- **Deep Learning Model**: TensorFlow/Keras CNN trained on ~87K images
- **38 Disease Classes**: Covers 14 different plant species
- **Multiple Input Methods**: 
  - Upload images (JPG, JPEG, PNG)
  - Live camera capture with front/rear camera support
- **Instant Analysis**: Real-time disease prediction with confidence levels
- **Comprehensive Treatment Information**:
  - Disease descriptions
  - Treatment recommendations
  - Prevention strategies
  - Agricultural best practices

### 🌤️ Weather Forecast
- **5-Day Forecast**:  Integrated OpenWeatherMap API
- **Agricultural Metrics**:
  - Temperature (current, high, low, feels-like)
  - Humidity levels (disease risk indicators)
  - Wind speed (spraying advisories)
  - Rainfall predictions
  - Cloud cover and atmospheric pressure
- **Smart Alerts**: Warnings for high humidity, strong winds, and significant rainfall
- **Location-Based**:  Search any location worldwide

### 🎨 User Interface
- **Responsive Design**: Built with Streamlit for seamless experience
- **Intuitive Navigation**: Sidebar with three main sections
- **Visual Feedback**: Progress indicators, success/warning messages, and balloons
- **Mobile-Friendly**: Camera integration for field use

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend/UI** | Streamlit 1.52.0+ |
| **Backend** | Python 3.8+ |
| **Machine Learning** | TensorFlow 2.20.0+, Keras |
| **Data Processing** | NumPy 2.3.5+, Pandas 2.3.3+, PIL (Pillow) 12.0.0+ |
| **Computer Vision** | OpenCV (opencv-python) |
| **API Integration** | Requests 2.32.5+, OpenWeatherMap API |
| **Security** | bcrypt, python-dotenv |
| **Camera Support** | streamlit_back_camera_input, streamlit_webrtc 0.64.5+ |
| **Development** | JupyterLab (for model training/analysis) |
| **Container Support** | Dev Container (Python 3.11) |

---

## 📂 Project Structure

```
Plant-disease-detection/
│
├── . devcontainer/
│   └── devcontainer.json          # VS Code Dev Container configuration
│
├── main.py                         # Main application file (Streamlit app)
├── trained_plant_disease_model.keras  # Pre-trained Keras model (~90 MB)
├── requirements.txt                # Python dependencies
├── . env                           # Environment variables (API keys)
├── home_page.jpeg                 # Application screenshot/banner
└── README.md                      # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenWeatherMap API key (free tier available)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sonukumar1722/Plant-disease-detection.git
   cd Plant-disease-detection
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the root directory:
   ```bash
   OPENWEATHERMAP_API_KEY=your_api_key_here
   ```
   
   Get your free API key from [OpenWeatherMap](https://openweathermap.org/api)

5. **Verify Model File**
   
   Ensure `trained_plant_disease_model. keras` is present in the root directory (included in repository)

6. **Run the Application**
   ```bash
   streamlit run main.py
   ```

7. **Access the Application**
   
   Open your browser and navigate to:  `http://localhost:8501`

---

## 📦 Dependencies

### Core Dependencies

```txt
streamlit>=1.52.0                   # Web application framework
tensorflow>=2.20.0                  # Deep learning framework
numpy>=2.3.5                        # Numerical computing
pandas>=2.3.3                       # Data manipulation
pillow>=12.0.0                      # Image processing
opencv-python                       # Computer vision
requests>=2.32.5                    # HTTP library for API calls
```

### Additional Components

```txt
streamlit_back_camera_input         # Camera input support
streamlit_webrtc>=0.64.5           # WebRTC for real-time streaming
ml-dtypes>=0.2.0                   # Machine learning data types
bcrypt                             # Password hashing (future auth)
python-dotenv                      # Environment variable management
jupyterlab                         # Model development environment
```

### Dependency Significance

| Package | Purpose | Why It's Essential |
|---------|---------|-------------------|
| **streamlit** | Web framework | Creates interactive UI without HTML/CSS/JS |
| **tensorflow** | ML framework | Loads and runs the trained CNN model |
| **numpy** | Array operations | Processes image data as arrays |
| **pillow** | Image handling | Opens, converts, and resizes images |
| **opencv-python** | Video processing | Enables camera input functionality |
| **requests** | API calls | Fetches weather data from OpenWeatherMap |
| **python-dotenv** | Config management | Securely loads API keys from .env file |

---

## 📄 File Descriptions

### 1. `main.py` (Core Application - 930 lines)

**Purpose**: Main Streamlit application containing all logic and UI

**Key Components**: 

#### a. **Configuration & Setup** (Lines 1-28)
- Import statements for all dependencies
- Page configuration with custom icon and layout
- Session state initialization
- Sidebar navigation setup

#### b. **Disease Classification Data** (Lines 29-44)
- `CLASS_NAMES`: List of 38 disease/healthy plant categories
- Supports 14 plant types:  Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

#### c. **Treatment Database** (Lines 46-568)
- `DISEASE_TREATMENTS`: Comprehensive dictionary with 38 entries
- Each entry contains:
  - **Description**: Disease symptoms and characteristics
  - **Treatment**: Step-by-step treatment protocols
  - **Prevention**:  Proactive measures to avoid disease
- Example structure:
  ```python
  'Apple___Apple_scab': {
      'description': 'Fungal disease causing.. .',
      'treatment': ['Apply fungicides... ', 'Remove fallen leaves...'],
      'prevention': ['Maintain proper spacing...', 'Avoid overhead irrigation...']
  }
  ```

#### d. **Model Management** (Lines 570-593)
- `load_model()`: Cached function to load Keras model (runs once)
- `model_prediction()`: 
  - Accepts image bytes or file path
  - Preprocesses image (resize to 128x128, RGB conversion)
  - Returns predicted class index

#### e. **Disease Prediction Page** (Lines 596-718)
- **Input Methods**:
  - File uploader with type validation
  - Camera input with preview
- **Prediction Logic**:
  - Real-time image analysis
  - Results display with color-coded status
  - Treatment and prevention recommendations
  - Special handling for healthy plants (balloons animation)

#### f. **Weather Forecast Page** (Lines 721-876)
- **API Integration**:
  - Geocoding to convert city names to coordinates
  - 5-day forecast retrieval
  - Data aggregation by day
- **Display Features**:
  - Gradient card design for each day
  - Weather icons and descriptions
  - Agricultural-specific metrics
  - Smart warnings (high humidity, strong winds, heavy rain)
  - Summary statistics

#### g. **Home & About Page** (Lines 879-929)
- Project overview and mission
- How-to guide
- Dataset information
- List of all 38 detectable conditions

**Significance**: 
- Central hub connecting ML model, API, and user interface
- Contains domain knowledge (treatment recommendations)
- Handles all user interactions and state management

---

### 2. `trained_plant_disease_model.keras` (~90 MB)

**Purpose**: Pre-trained Convolutional Neural Network model

**Specifications**:
- **Architecture**: CNN (Convolutional Neural Network)
- **Input Shape**: 128x128x3 (RGB images)
- **Output**:  38 classes (softmax layer)
- **Training Data**: ~70,295 images
- **Validation Data**: ~17,572 images
- **Format**: Keras native format (.keras)

**Training Details**:
- Data augmentation applied (rotation, flip, zoom)
- Trained on PlantVillage dataset
- Optimized for disease classification accuracy

**Significance**:  
- Core intelligence of the application
- Enables real-time disease detection
- Pre-trained for immediate deployment (no training needed)

---

### 3. `requirements.txt` (287 bytes)

**Purpose**: Python package dependencies specification

**Content**:
```txt
streamlit>=1.52.0 
streamlit_back_camera_input                            
streamlit_webrtc>=0.64.5
opencv-python                             
jupyterlab
tensorflow>=2.20.0
ml-dtypes>=0.2.0
numpy>=2.3.5
pandas>=2.3.3
pillow>=12.0.0
requests>=2.32.5
bcrypt
python-dotenv
```

**Significance**:
- Ensures reproducible environment
- Specifies minimum compatible versions
- Single-command installation (`pip install -r requirements.txt`)
- Version constraints prevent compatibility issues

---

### 4. `.env` (55 bytes)

**Purpose**: Environment variables storage (sensitive data)

**Content**:
```env
OPENWEATHERMAP_API_KEY=your_api_key_here
```

**Security Notes**:
- Should be added to `.gitignore` (keep private)
- Never commit API keys to version control
- Use `python-dotenv` to load variables securely

**Significance**:
- Separates configuration from code
- Enables different configs for dev/production
- Protects sensitive credentials

---

### 5. `.devcontainer/devcontainer.json` (1,016 bytes)

**Purpose**: VS Code Dev Container configuration

**Features**:
- **Base Image**: Python 3.11 on Debian Bookworm
- **Auto-Setup**: 
  - Installs dependencies on container creation
  - Runs Streamlit automatically on attach
- **Port Forwarding**: Port 8501 for web access
- **VS Code Extensions**:
  - Python extension
  - Pylance (language server)
- **GitHub Codespaces Ready**:  Opens README and main.py on launch

**Significance**: 
- One-click development environment
- Consistent setup across team members
- Cloud development support (Codespaces)

---

### 6. `home_page.jpeg` (75,864 bytes)

**Purpose**: Application screenshot or banner image

**Significance**:
- Visual documentation
- Marketing material
- GitHub repository preview

---

### 7. `README.md` (Current file)

**Purpose**: Project documentation

**Content**:
- Installation instructions
- Feature descriptions
- Technical specifications
- Usage guidelines

**Significance**:
- First point of contact for users
- Reduces support burden
- Improves project discoverability

---

## 📊 Dataset Information

### Source
- **Name**: PlantVillage Dataset
- **Link**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **License**: Public Domain

### Statistics
- **Total Images**: ~87,000 RGB images
- **Training Set**: 70,295 images (80%)
- **Validation Set**: 17,572 images (20%)
- **Test Set**: 33 images (evaluation)
- **Image Size**: 128x128 pixels (resized)
- **Classes**: 38 categories (diseases + healthy)

### Augmentation Techniques
- Rotation
- Width/Height shift
- Horizontal flip
- Zoom range
- Brightness adjustment

### Class Distribution
The dataset covers **14 plant species** with **multiple disease types** each:
- Apple (4 classes)
- Corn (4 classes)
- Grape (4 classes)
- Tomato (10 classes)
- Potato (3 classes)
- And 9 more species

---

## 🎯 Usage Guide

### Disease Prediction

1. **Navigate to Disease Prediction Page**
   - Use sidebar to select "Disease Prediction"

2. **Choose Input Method**
   - **Upload Image**: Click "Browse files" and select plant image
   - **Use Camera**: Click camera icon to capture live image

3. **Analyze Image**
   - Click "🔍 Predict Disease" button
   - Wait for analysis (typically 1-2 seconds)

4. **Review Results**
   - **Healthy Plants**: Green success message with prevention tips
   - **Diseased Plants**: 
     - Disease name and description
     - Step-by-step treatment protocol
     - Prevention strategies

### Weather Forecast

1. **Navigate to Weather Forecast Page**
   - Select "Weather Forecast" from sidebar

2. **Enter Location**
   - Type city name (e.g., "London", "New York", "Mumbai")
   - Click "Get Weather Forecast"

3. **Review Forecast**
   - 5-day weather cards with daily forecasts
   - Agricultural metrics (humidity, wind, rain)
   - Smart warnings for farming activities
   - Summary statistics at bottom

---

## 🌱 Supported Plants & Diseases

### Complete List (38 Classes)

| Plant | Diseases/Conditions |
|-------|---------------------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🧠 How It Works

### Disease Detection Pipeline

```
User Input → Image Preprocessing → CNN Model → Class Prediction → Result Display
    ↓             ↓                    ↓              ↓                ↓
 Upload/      Resize to          Load Keras      Softmax         Treatment
 Camera       128x128            Model           Output          Recommendations
             RGB Convert      (38 classes)    (Class Index)
```

### Model Architecture (Simplified)

1. **Input Layer**: 128x128x3 (RGB image)
2. **Convolutional Layers**: Feature extraction
3. **Pooling Layers**: Dimensionality reduction
4. **Dense Layers**: Classification
5. **Output Layer**: 38 neurons (softmax activation)

### Weather Forecast Pipeline

```
Location Input → Geocoding API → Lat/Lon → Weather API → Data Processing → Display
      ↓              ↓              ↓           ↓              ↓              ↓
   City Name    Convert to      Get         Fetch 5-day   Aggregate by   Format &
                Coordinates   Coordinates    Forecast        Day          Visualize
```

---

## 📈 Future Improvements

### Short-Term Enhancements
- [ ] Multi-language support (Hindi, Spanish, French)
- [ ] Offline mode with TensorFlow Lite
- [ ] Export prediction reports (PDF)
- [ ] User authentication and history tracking

### Medium-Term Goals
- [ ] Fertilizer recommendations based on soil type
- [ ] Pest identification module
- [ ] Integration with agricultural marketplaces
- [ ] Mobile app (React Native/Flutter)

### Long-Term Vision
- [ ] Real-time disease monitoring via IoT sensors
- [ ] AI chatbot for agricultural queries
- [ ] Drone integration for field analysis
- [ ] Community forum for farmers
- [ ] Machine learning model retraining with user data

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Reporting Bugs
1. Check existing issues to avoid duplicates
2. Create detailed bug report with: 
   - Steps to reproduce
   - Expected vs.  actual behavior
   - Screenshots (if applicable)
   - System information (OS, Python version)

### Suggesting Features
1. Open an issue with `[FEATURE]` tag
2. Describe the feature and its benefits
3. Provide use cases

### Submitting Pull Requests
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request with detailed description

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Update README if adding features

---

## 📧 Contact

**Project Maintainer**: [sonukumar1722](https://github.com/sonukumar1722)

**Repository**: [Plant-disease-detection](https://github.com/sonukumar1722/Plant-disease-detection)

For questions or support, please open an issue on GitHub. 

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - For providing comprehensive disease image data
- **OpenWeatherMap** - For weather API access
- **TensorFlow/Keras Team** - For the ML framework
- **Streamlit** - For the amazing web framework
- **Agricultural Research Community** - For domain knowledge and treatment protocols

---

## 📊 Project Statistics

![Repository Stats](https://img.shields.io/github/stars/sonukumar1722/Plant-disease-detection?style=social)
![Forks](https://img.shields.io/github/forks/sonukumar1722/Plant-disease-detection?style=social)
![Contributors](https://img.shields.io/github/contributors/sonukumar1722/Plant-disease-detection)

---

<div align="center">

**Made with ❤️ for farmers and agricultural enthusiasts**

*Protecting crops, one prediction at a time* 🌾

</div>
