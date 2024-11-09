from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import CNN
import io

app = FastAPI()

try:
    disease_df = pd.read_csv("disease_info.csv", encoding='ISO-8859-1')
except Exception as e:
    print(f"Error reading disease_info.csv: {e}")

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
    raise Exception("The column 'disease_name' is not found in the disease_info.csv file. Please check the file.")

try:
    supplement_df = pd.read_csv("supplement_info.csv", encoding='ISO-8859-1')
except Exception as e:
    print(f"Error reading supplements_info.csv: {e}")

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
    raise Exception("The column 'disease_name' is not found in the supplements_info.csv file. Please check the file.")

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

    return predicted_disease_name, disease_details, supplement_details

@app.post("/predict/")
async def get_disease_info(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    predicted_disease_name, disease_details, supplement_details = predict(image)

    response_data = {
        "disease_name": predicted_disease_name,
        "supplement_name": supplement_details.get('supplement_name', 'N/A'),
        "supplement_image": supplement_details.get('supplement_image', 'N/A'),
        "buy_link": supplement_details.get('buy_link', 'N/A')
    }

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
