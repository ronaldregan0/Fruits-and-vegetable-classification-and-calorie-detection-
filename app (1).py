from flask import Flask, render_template, request
import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import requests
from roboflow import Roboflow
app = Flask(__name__)

# Load the saved food classification and food freshness detection models
Food = load_model(r"model files\Indian_food_20.h5")
fre = load_model(r"model files\Food_freshness.h5")

# Load the PyTorch model for fruit and vegetable detection
model = AutoModelForImageClassification.from_pretrained(r"model files\Fruits_veggies_detection")
labels = list(model.config.id2label.values())  # Extracting labels from the model's configuration

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Your API key
API_KEY_nutrition = 'CMCYI9UOMNKkA+a5VVwtEw==1OqFgC2wwFxSEPC3'


# Menu and freshness options
menue = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']

fresh = ['rotten', 'fresh']

# Your Roboflow API key
rf = Roboflow(api_key="8RSJzoEweFB7NxxNK6fg")
# Your Roboflow model endpoint
PROJECT_NAME = 'fruit-veggies-od'

# Your Roboflow model endpoint
project = rf.workspace().project(PROJECT_NAME)
modelrf = project.version(1).model
def est4(test_image, roboflow):
    prediction = modelrf.predict(test_image, confidence=40, overlap=30).json()
    return prediction
# Function to perform food classification
def est2(test_image, model, labels):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=-1)
    return labels[pred_class[0]]

# Function to perform food freshness detection
def est3(test_image, model, labels):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img, (228, 228))
    img = img.reshape(1, 228, 228, 3)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=-1)
    confidence = np.max(prediction)
    return labels[pred_class[0]], confidence


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect_fruits_veggies', methods=['GET', 'POST'])
def detect_fruits_veggies():
    if request.method == 'POST':
        img_file = request.files['image']
        img_path = "static/" + img_file.filename
        img_file.save(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        input_tensor = preprocess(pil_image).unsqueeze(0)
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = labels[predicted_idx]
        return render_template("grocery.html", prediction=predicted_label, img_path=img_path)
    else:
        # Render the form if the request method is not 'POST'
        return render_template("grocery.html")

@app.route('/food_classification')
def food():
    return render_template('food_classification.html')

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = rimg_path = "static/" + img.filename
        img.save(img_path)
        prediction = est3(img_path, Food, menue)
        
        # Check if the prediction accuracy is above 65%
        if prediction[1] > 0.65:
            return render_template("food_classification.html", prediction=prediction[0], img_path=img_path)
        else:
            return render_template("food_classification.html", prediction="Model could not make a confident prediction", img_path=img_path)

@app.route('/freshness')
def freshness():
    return render_template('freshness.html')

@app.route("/submit3", methods=['POST'])
def get_output3():
    if request.method == 'POST':
        img = request.files['my_image3']
        img_path = "static/" + img.filename
        img.save(img_path)
        
        # Get the prediction from the Roboflow model
        result = modelrf.predict(img_path, confidence=40, overlap=30).json()
        
        # Extract the class from the prediction result
        if 'predictions' in result and len(result['predictions']) > 0:
            predicted_class = result['predictions'][0]['class']
        else:
            predicted_class = 'No prediction'
        
        # Render the template with the predicted class and image path
        return render_template("freshness.html", prediction=predicted_class, img_path=img_path)


@app.route('/get_nutrition_info', methods=['GET', 'POST'])
def get_nutrition_info():
    if request.method == 'POST':
        query = request.form.get('dish_name')
        
        # Updated API endpoint for CalorieNinjas
        api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
        response = requests.get(api_url + query, headers={'X-Api-Key': API_KEY_nutrition})
        print("API URL:", api_url + query)  # Print the API URL for debugging
        print("API Response Code:", response.status_code)  # Print the response code for debugging
        print("API Response Text:", response.text)  # Print the response text for debugging
        
        if response.status_code == requests.codes.ok:
            nutrition_info = response.json()
            print("API Response:", nutrition_info)  # Print the API response
            
            # The CalorieNinjas API returns data in 'items' array
            if 'items' in nutrition_info and nutrition_info['items']:
                return render_template('nutrition_info.html', nutrition_info=nutrition_info['items'])
            else:
                error_message = 'Error: No nutrition information available for "{}".'.format(query)
                return render_template('nutrition_info.html', error=error_message)
        else:
            error_message = 'Error: Unable to fetch nutrition information. Please try again.'
            return render_template('nutrition_info.html', error=error_message)
    else:
        # Handle GET request (render a form or provide alternative content)
        return render_template('nutrition_info.html')

@app.route('/diet')
def faq():
    return render_template('faq.html')

if __name__ == '__main__':
    app.run(debug=True)
