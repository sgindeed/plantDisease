# 🌿 Plant Disease Detector

[![Streamlit Deployment](https://img.shields.io/badge/Deployed%20on-Streamlit-brightgreen)](https://plantdisease-greenhack.streamlit.app/)

## 🌱 Project Overview
**Plant Disease Detector** is a FastAPI-based web app that helps identify plant diseases from uploaded images. Using a pre-trained CNN model, it diagnoses diseases, provides details and preventive measures, and suggests relevant supplements.

## 🔍 Features
- **Disease Detection** 🧪: Upload a plant image to receive a diagnosis and disease information.
- **Prevention Advice** 🛡️: View prevention steps for identified diseases.
- **Supplement Recommendations** 🌿: Suggested supplements with direct buy links for managing plant health.
- **API-based**: Built with FastAPI for high performance and scalability.

## 📁 Project Structure

```
plantDisease/
├── app.py                  # Main FastAPI application file
├── CNN.py                  # CNN model definition
├── disease_info.csv        # CSV containing disease names, descriptions, images, and prevention tips
├── supplement_info.csv     # CSV containing supplement names, images, and purchase links
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- **Python 3.x** 🐍
- Libraries listed in `requirements.txt` 📋

### Installation

1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/sgindeed/plantDisease.git
   cd plantDisease
   ```

2. **Install Dependencies** 📦:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application** 🏃:
   ```bash
   uvicorn app:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.

### Deployment on Streamlit
This app is also deployed on Streamlit for quick access! [Check it out here](https://plantdisease-greenhack.streamlit.app/) 🌐.

## 🖼️ Usage

1. **Home Page** 🏡: Upload an image of the affected plant.
2. **Prediction Output** 🔍:
   - **Disease Name**: The detected disease’s name.
   - **Description**: Details about the disease.
   - **Prevention**: Suggested measures to prevent or manage the disease.
3. **Supplements** 🌿:
   - Recommended supplements with images and purchase links.

## 🧠 Model Details
The app uses a **Convolutional Neural Network (CNN)** model trained to classify 39 plant diseases. The model processes the uploaded image and returns the best match along with relevant disease information.

### Model Pipeline
1. **Image Preprocessing** 🖼️: Resizes and prepares the image for the model.
2. **Disease Prediction** 🧬: Passes the image through the CNN model.
3. **Data Retrieval** 📖: Displays disease information and supplements.

## 📊 Technology Stack
- **Backend**: FastAPI 🚀
- **Model**: PyTorch (CNN-based) 🧠
- **Image Processing**: PIL 🖼️
- **Frontend**: Basic HTML for upload interface 📄
- **Deployment**: Streamlit 🌐

## 🤝 Contributions
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository 🍴.
2. Create a branch (`feature/YourFeatureName`) 🌿.
3. Commit your changes 💾.
4. Open a Pull Request 🚀.

## 📬 Contact
For questions or feedback, reach out to [@sgindeed](https://github.com/sgindeed) on GitHub.
