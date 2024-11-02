import os
from flask import Flask, request, jsonify
from PIL import Image
import CNN
import numpy as np
import torch
import pandas as pd
import torchvision.transforms.functional as TF


# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the CNN model
model = CNN.CNN(39)
model.load_state_dict(torch.load("G:\\plant disease\\plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

# Home Page
@app.route('/')
def home_page():
    return "Welcome to Plant Disease Detector. Use /submit to upload an image."

# Submit Image Page
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        response = {
            "title": title,
            "description": description,
            "prevention": prevent,
            "disease_image": image_url,
            "supplement_name": supplement_name,
            "supplement_image": supplement_image_url,
            "buy_link": supplement_buy_link
        }
        return jsonify(response)
    
    return '''<form action="/submit" method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Submit</button>
              </form>'''

# Market Page
@app.route('/market')
def market():
    supplements = []
    for name, image, link in zip(supplement_info['supplement name'], supplement_info['supplement image'], supplement_info['buy link']):
        supplements.append({
            "name": name,
            "image": image,
            "buy_link": link
        })
    return jsonify(supplements)

if __name__ == '__main__':
    app.run(debug=True)
