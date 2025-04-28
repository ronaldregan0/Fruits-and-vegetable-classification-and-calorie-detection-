from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from transformers import AutoModelForImageClassification
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import requests
from roboflow import Roboflow
import io
import tempfile
import google.generativeai as genai
from transformers import AutoImageProcessor, AutoModelForImageClassification
from inference_sdk import InferenceHTTPClient
import json
from fastapi.middleware.cors import CORSMiddleware



genai.configure(api_key="AIzaSyArFsF8XTEyuPDbQhtvGjZfygziLN6RF7o")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize the FastAPI app


# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specify allowed origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved models
Food = load_model("model files/Indian_food_20.h5")
fre = load_model("model files/Food_freshness.h5")

# Load the PyTorch model for fruit and vegetable detection
model = AutoModelForImageClassification.from_pretrained("model files/Fruits_veggies_detection")
labels = list(model.config.id2label.values())  # Extracting labels from the model's configuration

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Roboflow setup for fruit and vegetable detection
rf = Roboflow(api_key="8RSJzoEweFB7NxxNK6fg")
PROJECT_NAME = 'fruit-veggies-od'
project = rf.workspace().project(PROJECT_NAME)
modelrf = project.version(1).model

# Define menu and freshness options
menue = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']
fresh = ['rotten', 'fresh']

# Define the models' inference functions
def est2(test_image, model, labels):
    img = cv2.imdecode(np.frombuffer(test_image, np.uint8), -1)  # Decode image from bytes
    img = img / 255.0
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=-1)
    return labels[pred_class[0]]

def est3(test_image):
    processor = AutoImageProcessor.from_pretrained("dima806/indian_food_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/indian_food_image_detection")
    img = Image.open(io.BytesIO(test_image))
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = probabilities.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item()

    return predicted_label, confidence

def est4(test_image, roboflow):
    prediction = modelrf.predict(test_image, confidence=40, overlap=30).json()
    return prediction

# API Endpoints
@app.post("/detect_fruits_veggies")
async def detect_fruits_veggies(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess(pil_image).unsqueeze(0)
    outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = labels[predicted_idx]
    return JSONResponse(content={"prediction": predicted_label})

@app.post("/food_classification")
async def food_classification(medical_conditions: list[str], image: UploadFile = File(...)):
    image_bytes = await image.read()
    prediction_ = est3(image_bytes)
    response = None
    CLIENT = InferenceHTTPClient( 
            api_url="https://detect.roboflow.com",
            api_key="3AllLtv6oRaUdJGMUdek"
        )
    img = Image.open(io.BytesIO(image_bytes))
    result = CLIENT.infer(img, model_id="south-indian-food-detection/3")
    if result:
        print(result)
        food = []
        conf = []  
        
        for prediction in result["predictions"]:
            food.append(prediction["class"])
            conf.append(prediction["confidence"])
        result = " & ".join(food)
        if medical_conditions:
            prompt = f"Based on the given medical conditions, tell me whether the person should b eating {result} for food or not.\nMedical Conditions: {medical_conditions}\nMake sure to reply like \"You should not be eating that\" or \"You can eat that\""
            response = gen_model.generate_content(prompt)
        if max(conf)>0.8:
            return JSONResponse(content={"prediction": result, "suggestion": response.text})
    if medical_conditions:
        prompt = f"Based on the given medical conditions, tell me whether the person should b eating {prediction_[0]} for food or not.\nMedical Conditions: {medical_conditions}\nMake sure to reply like \"You should not be eating that\" or \"You can eat that\""
        response = gen_model.generate_content(prompt)
        return JSONResponse(content={"prediction": prediction_[0], "suggestion": response.text})

@app.post("/freshness")
async def freshness(image: UploadFile = File(...)):
    image_bytes = await image.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
        tmpfile.write(image_bytes)
        tmpfile_path = tmpfile.name
        result = modelrf.predict(tmpfile_path, confidence=40, overlap=30).json()
    
    if 'predictions' in result and len(result['predictions']) > 0:
        predicted_class = result['predictions'][0]['class']
    else:
        predicted_class = 'No prediction'

    return JSONResponse(content={"prediction": predicted_class})

@app.post("/get_nutrition_info")
async def get_nutrition_info(dish_name: str):
    API_KEY_nutrition = 'CMCYI9UOMNKkA+a5VVwtEw==1OqFgC2wwFxSEPC3'
    api_url = f'https://api.calorieninjas.com/v1/nutrition?query={dish_name}'
    headers = {'X-Api-Key': API_KEY_nutrition}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        nutrition_info = response.json()
        if 'items' in nutrition_info and nutrition_info['items']:
            return JSONResponse(content={"nutrition_info": nutrition_info['items']})
        else:
            prompt = f'''Give nutriotional information about {dish_name} in the format
            \"name\": ,
            \"calories\": ,
            \"serving_size_g\": ,
            \"fat_total_g\": ,
            \"fat_saturated_g\": ,
            \"protein_g\": ,
            \"sodium_mg\": ,
            \"potassium_mg\": ,
            \"cholesterol_mg\": ,
            \"carbohydrates_total_g\": ,
            \"fiber_g\": ,
            \"sugar_g\": 
            Make sure to have all items comma separated and give no other text'''
            response = gen_model.generate_content(prompt)
            data = json.loads("{"+response.text+"}")
            return JSONResponse(content={"nutrition_info": data})
    else:
        raise HTTPException(status_code=response.status_code, detail="Unable to fetch nutrition information")

class UserProfile(BaseModel):
    height: float = Field(..., description="Height of the user in cm")
    weight: float = Field(..., description="Weight of the user in kg")
    age: int = Field(..., ge=1, le=120, description="Age of the user in years")
    food_preference: str = Field(..., description="Food preference of the user (e.g., veg, non-veg, vegan, keto, etc.)")
    medical_conditions: list[str] = Field([], description="Optional list of medical conditions (e.g., diabetes, hypertension, etc.)")

@app.post("/get_custom_diet")
async def get_custom_diet(user_profile: UserProfile):
    # Prepare input text for the Gemini API based on the user's profile
    profile_text = f"User Profile:\n"
    profile_text += f"Height: {user_profile.height} cm\n"
    profile_text += f"Weight: {user_profile.weight} kg\n"
    profile_text += f"Age: {user_profile.age} years\n"
    profile_text += f"Food Preference: {user_profile.food_preference}\n"
    
    if user_profile.medical_conditions:
        profile_text += f"Medical Conditions: {', '.join(user_profile.medical_conditions)}\n"
    else:
        profile_text += "Medical Conditions: None\n"

    prompt = f"Given the following user profile, suggest a detailed custom diet chart:\n{profile_text}"

    # Send the request to Gemini for a custom diet chart
    try:
        response = gen_model.generate_content(prompt)
        diet_chart = response.text
        return JSONResponse(content={"custom_diet_chart": diet_chart})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diet chart: {str(e)}")
