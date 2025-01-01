# Plant Disease Prediction System

## Project Overview
The Plant Disease Prediction System is an AI-powered application designed to detect and classify plant diseases from images of leaves. By leveraging machine learning and computer vision techniques, the system provides farmers with an efficient tool for early disease detection, enabling timely interventions to minimize crop losses and ensure better agricultural productivity.

---

## Features
- **Real-Time Disease Detection:** Upload a leaf image and get instant disease classification.
- **Disease Information:** Detailed descriptions and symptoms for each detected disease.
- **Preventive Measures:** Suggestions for treatment and prevention.
- **User-Friendly Interface:** Accessible via a web-based UI built with Streamlit.
- **Multi-Language Support:** Enhances accessibility for rural and non-English speaking users.

---

## System Architecture
1. **Data Collection:** Diverse dataset of healthy and diseased leaf images.
2. **Preprocessing:** Image normalization and data augmentation for robustness.
3. **Model Training:** Transfer learning with a Convolutional Neural Network (e.g., ResNet50).
4. **Prediction:** Confidence scores and disease classification.
5. **User Interface:** Web-based application for ease of use.

---

## Technology Stack
- **Programming Language:** Python
- **Frameworks:** TensorFlow, Keras
- **Web Interface:** Streamlit
- **Deployment:** Compatible with cloud platforms or local environments

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd plant-disease-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Dataset
The model is trained on the PlantVillage dataset, supplemented with additional images to ensure diverse crop and disease representation. 

---

## Results
- **Accuracy:** 95% on the testing dataset
- **Evaluation Metrics:** High precision, recall, and F1-scores for most disease categories.

---

## Future Enhancements
- Expand the dataset to cover more crops and diseases.
- Integrate IoT devices for real-time monitoring.
- Develop a mobile app for wider accessibility.
- Incorporate weather and soil condition data for a holistic approach.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

