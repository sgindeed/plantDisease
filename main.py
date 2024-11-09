import streamlit as st
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import CNN

try:
    disease_df = pd.read_csv("disease_info.csv", encoding='ISO-8859-1')
except Exception as e:
    st.error(f"Error reading disease_info.csv: {e}")

if 'disease_name' in disease_df.columns:
    disease_info = {
        row['disease_name']: {
            "description": row['description'],
            "possible_steps": row['Possible Steps'],
            "image_url": row['image_url'],
        }
        for _, row in disease_df.iterrows()
    }
else:
    st.error("The column 'disease_name' is not found in the disease_info.csv file. Please check the file.")

try:
    supplement_df = pd.read_csv("supplement_info.csv", encoding='ISO-8859-1')
except Exception as e:
    st.error(f"Error reading supplements_info.csv: {e}")

if 'disease_name' in supplement_df.columns:
    supplement_info = {
        row['disease_name']: {
            "supplement_name": row['supplement name'],
            "supplement_image": row['supplement image'],
            "buy_link": row['buy link']
        }
        for _, row in supplement_df.iterrows()
    }
else:
    st.error("The column 'disease_name' is not found in the supplements_info.csv file. Please check the file.")

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_disease_name = list(disease_info.keys())[predicted.item()]
    disease_details = disease_info.get(predicted_disease_name, {})
    supplement_details = supplement_info.get(predicted_disease_name, {})

    return predicted.item(), predicted_disease_name, disease_details, supplement_details

st.title("Plant Disease Detection")
st.write("Upload an image of the plant leaf to check for diseases.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict"):
        prediction, predicted_disease_name, disease_details, supplement_details = predict(image)

        st.write(f"Predicted Disease Class: {prediction}")

        if disease_details:
            st.subheader(predicted_disease_name)
            st.write(f"Description: {disease_details['description']}")
            st.write(f"Possible Steps: {disease_details['possible_steps']}")
            st.image(disease_details["image_url"], caption='Disease Image', use_column_width=True)

            if supplement_details:
                st.write(f"Recommended Supplement: {supplement_details['supplement_name']}")
                st.image(supplement_details["supplement_image"], caption='Supplement Image', use_column_width=True)
                st.markdown(f"[Buy here]({supplement_details['buy_link']})")
            else:
                st.write("No supplement information available for this disease.")
        else:
            st.write("No details found for the predicted disease.")

st.markdown("""
    <footer style="margin-top: 20px; font-size: 14px; text-align: center; color: #555;">
        Made with ❤️ and ⚡ by Supratim
    </footer>
""", unsafe_allow_html=True)
