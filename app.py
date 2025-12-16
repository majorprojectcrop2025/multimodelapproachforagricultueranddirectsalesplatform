import os
import pickle
import threading
import webbrowser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from PIL import Image
from torchvision import models, transforms
import google.generativeai as genai
import sqlite3
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore
except ImportError:
    EfficientNet = None
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, 'site.db')

# Paths to direct sales data
FARMER_TO_CONSUMER_CSV = os.path.join(BASE_DIR, "direct sales data", "farmersyoconsemers", "Farmer_to_Consumer_Karnataka_50.csv")
CONSUMER_REQUEST_CSV = os.path.join(BASE_DIR, "direct sales data", "farmersyoconsemers", "Consumer_Request_Dataset_50.csv")
FARMER_TO_APARTMENT_CSV = os.path.join(BASE_DIR, "direct sales data", "farmerstoapartment", "FarmerToApartmentDataset.csv")
APARTMENT_CROP_DEMAND_CSV = os.path.join(BASE_DIR, "direct sales data", "farmerstoapartment", "ApartmentCropDemand.csv")
# New paths for consumer-centric feature
CONSUMER_CROP_REQUEST_DATASET_CSV = os.path.join(BASE_DIR, "direct sales data", "consumersrequestdata", "ConsumerCropRequestDataset.csv")
CONSUMER_FARMER_DATASET_CSV = os.path.join(BASE_DIR, "direct sales data", "consumersrequestdata", "FarmerDataset.csv")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"✅ Gemini model '{GEMINI_MODEL_NAME}' initialized.")
    except Exception as exc:
        print(f"⚠️ Failed to initialize Gemini model: {exc}")
else:
    print("⚠️ GEMINI_API_KEY not set; agri chatbot endpoint will be disabled.")

# Database functions
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


# NEW YIELD PREDICTION MODELS
NEW_YIELD_MODELS_BASE_DIR = os.path.join(BASE_DIR, "newyeildprediction")
new_label_encoders = pickle.load(open(os.path.join(NEW_YIELD_MODELS_BASE_DIR, "label_encoders.pkl"), "rb"))
production_rf_model = pickle.load(open(os.path.join(NEW_YIELD_MODELS_BASE_DIR, "production_rf_model.pkl"), "rb"))

# Karnataka yield model
KARNATAKA_MODEL_PATH = os.path.join(BASE_DIR, "traindtranatakadata", "yield_prediction_model.pkl")
with open(KARNATAKA_MODEL_PATH, "rb") as f:
    karnataka_yield_model, karnataka_label_encoders = pickle.load(f)
karnataka_feature_order = getattr(
    karnataka_yield_model,
    "feature_names_in_",
    np.array(["District", "crop", "croptype", "Area", "Production"]),
)
karnataka_categorical_options = {
    col: sorted([str(value) for value in encoder.classes_])
    for col, encoder in karnataka_label_encoders.items()
}

def preprocess_new_yield_input(user_data):
    # Encode categorical columns for new yield prediction
    for col, le in new_label_encoders.items():
        try:
            user_data[col] = le.transform([user_data[col]])[0]
        except ValueError:
            raise ValueError(f"Unseen label for {col}: '{user_data[col]}'. Please provide a valid input.")
    
    # Order of features for the new model
    input_list = [
        user_data["State_Name"],
        user_data["N"],
        user_data["P"],
        user_data["K"],
        user_data["pH"],
        user_data["rainfall"],
        user_data["temperature"],
        user_data["Are-in_hectares"],
        user_data["Crop_Type"],
        user_data["Crop"]
    ]
    processed_array = np.array([input_list])
    print(f"Shape of processed_array: {processed_array.shape}")
    return processed_array

def predict_new_yield(user_data):
    try:
        processed_input = preprocess_new_yield_input(user_data)
        print(f"Shape of processed_input before prediction: {processed_input.shape}")
        prediction = production_rf_model.predict(processed_input)[0]
        # Assuming the model predicts in some unit, convert to tons if necessary (e.g., if it's kg)
        # For now, let's assume the model's output is directly the yield, and we just need to display it.
        # If the model predicts in kg and you want tons, divide by 1000: prediction_in_tons = prediction / 1000
        return prediction
    except ValueError as e:
        raise e
    except Exception as e:
        print(f"Error during new yield prediction: {e}")
        raise Exception("An unexpected error occurred during new yield prediction.")


# --------------------------
# Flask setup
# --------------------------

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = 'your_super_secret_key_here' # Replace with a strong, random key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lemon crop disease detector config
LEMON_MODEL_PATH = os.path.join(BASE_DIR, "pakacropdiseases", "lemon", "mobilenet_lemon_disease_model.pth")
LEMON_CLASS_NAMES = [
    "Alternaria Leaf Spot",
    "Anthracnose",
    "Canker",
    "Citrus Scab",
    "Greasy Spot",
    "Healthy",
    "Leaf Miner",
    "Nutrient Deficiency",
    "Sooty Mold",
]
lemon_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
lemon_model = None
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "pakacropdiseases", "potato", "mobilenetv2_potato_disease.pth")
POTATO_CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy",
    "Bacterial Wilt",
    "Early Rot",
    "Phosphorus Deficiency",
]
potato_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
potato_model = None
RICE_MODEL_PATH = os.path.join(BASE_DIR, "pakacropdiseases", "rice", "resnet50_rice_leaf_disease.pth")
RICE_CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Blast",
    "Neck Blast",
    "Sheath Blight",
    "Leaf Smut",
    "Rice Hispa",
    "Rice Tungro",
    "Blast",
    "Healthy",
]
rice_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
rice_model = None
TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "pakacropdiseases", "tomato", "resnet50_tomato_leaf_diseases.pth")
TOMATO_CLASS_NAMES = [
    'Tomato___Bacterial_spot_diseasede',
    'Tomato___Early_blight_diseasede',
    'Tomato___Late_blight_diseasede',
    'Tomato___Target_Spot_diseasede',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus_diseasede',
    'Tomato___Tomato_mosaic_virus_diseasede',
    'Tomato___healthy_healthy'
]
tomato_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tomato_model = None
CUCUMBER_MODEL_PATH = os.path.join(BASE_DIR, "pakacropdiseases", "cucumber", "resnet50_cucumber_disease.pth")
CUCUMBER_CLASS_NAMES = [
    "Anthracnose",
    "Downy Mildew",
    "Fusarium Wilt",
    "Gummy Stem Blight",
    "Leaf Spot",
    "Powdery Mildew",
    "Target Spot",
    "Healthy",
]
cucumber_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
cucumber_model = None

# --------------------------
# Configurations
# --------------------------

num_classes = 38

class_names = [
    'Apple_scab_diseasede', 'Apple_Black_rot_diseasede', 'Apple___Cedar_apple_rust_diseasede', 'Apple___healthy',

    'Blueberry___healthy', 'Cherry_Powdery_mildew_diseasede', 'Cherry___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot_diseasede', 'Corn_(maize)___Common_rust__diseasede',

    'Corn_(maize)___Northern_Leaf_Blight_diseasede', 'Corn_(maize)___healthy_healthy', 'Grape___Black_rot_diseasede',

    'Grape___Esca_(Black_Measles)_diseasede', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)_diseasede', 'Grape___healthy',

    'Orange___Haunglongbing_(Citrus_greening)_diseasede', 'Peach___Bacterial_spot_diseasede', 'Peach___healthy',

    'Pepper,_bell___Bacterial_spot_diseasede', 'Pepper,_bell___healthy', 'Potato___Early_blight_diseasede',

    'Potato___Late_blight_diseasede', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',

    'Squash___Powdery_mildew_diseasede', 'Strawberry___Leaf_scorch_diseasede', 'Strawberry___healthy',

    'Tomato___Bacterial_spot_diseasede', 'Tomato___Early_blight_diseasede', 'Tomato___Late_blight_diseasede', 'Tomato___Leaf_Mold_diseasede',

    'Tomato___Septoria_leaf_spot_diseasede', 'Tomato___Spider_mites Two-spotted_spider_mite_diseasede', 'Tomato___Target_Spot_diseasede',

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus_diseasede', 'Tomato___Tomato_mosaic_virus_diseasede', 'Tomato___healthy_healthy'
]

# Transformations (same as during training)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --------------------------
# Load all models
# --------------------------

def load_models():

    models_list = []

    # ResNet50

    resnet50 = models.resnet50(pretrained=False)

    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

    resnet50.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models/plantdiseasemodel/resnet50_plant_disease_detector.pth"),
        map_location=device
    ))

    resnet50.to(device).eval()

    models_list.append(("ResNet50", resnet50))

    # DenseNet201

    densenet = models.densenet121(pretrained=False)

    densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, num_classes)

    densenet.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models/plantdiseasemodel/densenet_plant_disease_classifier.pth"),
        map_location=device
    ))

    densenet.to(device).eval()

    models_list.append(("DenseNet121", densenet))

    # EfficientNet-B3

    effnet = models.efficientnet_b0(pretrained=False)

    effnet.classifier[1] = torch.nn.Linear(effnet.classifier[1].in_features, num_classes)

    effnet.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models/plantdiseasemodel/efficientnet_plant_disease_model.pth"),
        map_location=device
    ))

    effnet.to(device).eval()

    models_list.append(("EfficientNetB3", effnet))

    # MobileNetV3

    mobilenet = models.mobilenet_v2(pretrained=False)

    mobilenet.classifier[1] = torch.nn.Linear(mobilenet.classifier[1].in_features, num_classes)

    mobilenet.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models/plantdiseasemodel/mobilenet_v2_plant_diseases.pth"),
        map_location=device
    ))

    mobilenet.to(device).eval()

    models_list.append(("MobileNetV3", mobilenet))

    return models_list

models_list = load_models()

# Base path for yield prediction models
YIELD_MODELS_BASE_DIR = os.path.join(BASE_DIR, "olderyeildpredicton")

# Load yield prediction models
knn = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "knn_model (1).pkl"), "rb"))
xgb = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "xgb_model (1).pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "kmeans_model (1).pkl"), "rb"))

# Load scaler & label encoders
scaler = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "scaler (1).pkl"), "rb"))
label_encoders = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "label_encoders (1).pkl"), "rb"))

# Load cluster mean yield mapping
cluster_mean_yield = pickle.load(open(os.path.join(YIELD_MODELS_BASE_DIR, "cluster_mean_yield (1).pkl"), "rb"))

# Extract categorical options for dropdowns
categorical_options = {
    col: sorted([str(x) for x in le.classes_]) for col, le in label_encoders.items()
}

DISEASE_INFO_PATH = os.path.join(BASE_DIR, "datavcfarm", "vcfarm_disease_info.csv")
disease_df = pd.read_csv(DISEASE_INFO_PATH)

def get_disease_info(disease_name):
    row = disease_df[disease_df["disease_name"] == disease_name]
    if row.empty:
        return None # Return None if disease not found
    row = row.iloc[0]
    return {
        "disease_name": row["disease_name"],
        "description": row["description"],
        "steps": row["Possible Steps"] if "Possible Steps" in row.index else "N/A", # Use 'Possible Steps' for key
        "image_url": row["image_url"] if "image_url" in row.index else None
    }

# --------------------------
# Ensemble prediction (Majority Voting)
# --------------------------

def ensemble_predict(image_path, confidence_threshold=0.60):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    results = {}

    votes = []

    confidences = defaultdict(list)

    # Initialize a tensor to accumulate probabilities for soft voting
    # The size should be (1, num_classes)
    soft_votes = torch.zeros(1, num_classes).to(device)

    # Get predictions (probabilities) from all models
    for name, model in models_list:
        with torch.no_grad():
            output = model(img_tensor)
            prob = F.softmax(output, dim=1) # Get probabilities
            soft_votes += prob # Accumulate probabilities

            # Also store individual model results for detailed view
            pred_idx = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred_idx].item()
            predicted_class = class_names[pred_idx]
            results[name] = (predicted_class, confidence)

            # For hard voting
            votes.append(predicted_class)
            confidences[predicted_class].append(confidence)

    # Perform soft voting by averaging the probabilities
    avg_probs = soft_votes / len(models_list)
    
    # Get the class with the highest average probability
    score, pred_idx = torch.max(avg_probs, dim=1)
    ensemble_confidence = score.item()
    ensemble_predicted_class = class_names[pred_idx.item()]

    # Check if confidence is above threshold
    if ensemble_confidence < confidence_threshold:
        final_prediction_message = "Unknown class – result not found"
        results["Ensemble (Soft Voting)"] = (final_prediction_message, ensemble_confidence)
    else:
        results["Ensemble (Soft Voting)"] = (ensemble_predicted_class, ensemble_confidence)

    # Hard Voting (Majority Voting)
    vote_counts = Counter(votes)
    hard_vote_class, hard_vote_count = vote_counts.most_common(1)[0]
    hard_vote_confidence = sum(confidences[hard_vote_class]) / len(confidences[hard_vote_class]) if confidences[hard_vote_class] else 0.0

    if hard_vote_confidence < confidence_threshold:
        hard_vote_message = "Unknown class – result not found"
        results["Ensemble (Hard Voting)"] = (hard_vote_message, hard_vote_confidence)
    else:
        results["Ensemble (Hard Voting)"] = (hard_vote_class, hard_vote_confidence)

    return results

# --------------------------
# Yield Prediction Functions
# --------------------------

def preprocess_input(user_data):
    # user_data is a dictionary

    # Encode categorical columns
    for col, le in label_encoders.items():
        try:
            user_data[col] = le.transform([user_data[col]])[0]
        except ValueError:
            raise ValueError(f"Unseen label for {col}: '{user_data[col]}'. Please provide a valid input.")

    # Convert to list
    input_list = [
        user_data["Location"],
        user_data["Area"],
        user_data["Rainfall"],
        user_data["Temperature"],
        user_data["Soil type"],
        user_data["Irrigation"],
        user_data["Humidity"],
        user_data["Crops"],
        user_data["price"],
        user_data["Season"]
    ]

    # Scale
    input_scaled = scaler.transform([input_list])

    return input_scaled

def predict_ensemble(user_data):

    try:
        X_scaled = preprocess_input(user_data)
    except ValueError as e:
        raise e # Re-raise the error to be caught by the route handler

    # 1. KNN prediction
    knn_pred = knn.predict(X_scaled)[0]

    # 2. XGBoost prediction
    xgb_pred = xgb.predict(X_scaled)[0]

    # 3. K-Means cluster prediction
    cluster = kmeans.predict(X_scaled)[0]
    # Assuming cluster_mean_yield maps directly from cluster ID to mean yield
    cluster_pred = cluster_mean_yield[cluster]

    # Average ensemble
    final_yield = (knn_pred + xgb_pred + cluster_pred) / 3

    return {
        "knn": knn_pred,
        "xgboost": xgb_pred,
        "kmeans_cluster": cluster_pred,
        "final_ensemble_yield": final_yield
    }

# --------------------------
# Flask routes
# --------------------------

@app.route("/")
def main_page():
    return render_template("main.html")

@app.route("/disease_detection", methods=["GET", "POST"])
def disease_detection():
    result = None
    uploaded_filename = None
    error_message = None

    if request.method == "POST":
        # Check if a file was uploaded
        if 'uploaded_image' not in request.files:
            error_message = "No image file part in the request."
        else:
            file = request.files['uploaded_image']

            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                error_message = "No selected image file."
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "disease_detection", filename)

                # Perform image validation (plant vs. non-plant) is now handled by ensemble model
                # is_plant_image = validate_plant_image(filepath) # Removed

                # if not is_plant_image: # Removed
                #     error_message = "Invalid image. Please upload a plant disease image." # Removed
                #     os.remove(filepath) # Removed
                # else: # Removed
                try:
                    confidence_threshold = 0.60
                    result = ensemble_predict(filepath, confidence_threshold)
                except Exception as e:
                    print(f"Error during disease prediction: {e}")
                    error_message = "An unexpected error occurred during disease prediction."
            else:
                error_message = "Allowed image types are png, jpg, jpeg, gif."

    return render_template("disease_detection.html", result=result, filename=uploaded_filename, error_message=error_message)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/yield_prediction", methods=["GET", "POST"])
def yield_prediction_page():
    yield_predictions = None
    yield_error = None

    if request.method == "POST":
        try:
            user_data = {
                "Location": request.form["location"],
                "Area": float(request.form["area"]),
                "Rainfall": float(request.form["rainfall"]),
                "Temperature": float(request.form["temperature"]),
                "Soil type": request.form["soil_type"],
                "Irrigation": request.form["irrigation"],
                "Humidity": float(request.form["humidity"]),
                "Crops": request.form["crops"],
                "price": float(request.form["price"]),
                "Season": request.form["season"]
            }
            yield_predictions = predict_ensemble(user_data)
        except ValueError as e:
            yield_error = str(e)
        except Exception as e:
            print(f"Error during yield prediction: {e}")
            yield_error = "An unexpected error occurred during yield prediction."

    return render_template("yield_prediction.html", yield_predictions=yield_predictions, yield_error=yield_error, categorical_options=categorical_options)

@app.route("/disease_description", methods=["GET", "POST"])
def disease_description_page():
    disease_names = sorted(disease_df["disease_name"].unique().tolist())
    disease_info = None

    if request.method == "POST":
        selected_disease = request.form["disease_name"]
        disease_info = get_disease_info(selected_disease)

    return render_template("disease_description.html", disease_names=disease_names, disease_info=disease_info)

@app.route("/new_yield_prediction", methods=["GET", "POST"])
def new_yield_prediction_page():
    new_yield_predictions = None
    new_yield_error = None
    
    # Extract categorical options for the new model's dropdowns
    new_categorical_options = {
        col: sorted([str(x) for x in le.classes_]) for col, le in new_label_encoders.items()
    }

    if request.method == "POST":
        try:
            user_data = {
                "State_Name": request.form["state_name"],
                "N": float(request.form["N"]),
                "P": float(request.form["P"]),
                "K": float(request.form["K"]),
                "pH": float(request.form["pH"]),
                "rainfall": float(request.form["rainfall"]),
                "temperature": float(request.form["temperature"]),
                "Are-in_hectares": float(request.form["are-in_hectares"]),
                "Crop_Type": request.form["crop_type"],
                "Crop": request.form["crop"]
            }
            new_yield_predictions = predict_new_yield(user_data)
        except ValueError as e:
            new_yield_error = str(e)
        except Exception as e:
            print(f"Error during new yield prediction: {e}")
            new_yield_error = "An unexpected error occurred during new yield prediction."

    return render_template("new_yield_prediction.html", new_yield_predictions=new_yield_predictions, new_yield_error=new_yield_error, new_categorical_options=new_categorical_options)


@app.route("/agri_chatbot")
def agri_chatbot_page():
    return render_template("agri_chatbot.html", gemini_enabled=gemini_model is not None)


@app.route("/api/agri_chat", methods=["POST"])
def agri_chat_api():
    if gemini_model is None:
        return jsonify(
            {"status": "error", "error": "Gemini API key not configured on server."}
        ), 500

    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return jsonify({"status": "error", "error": "Message cannot be empty."}), 400

    prompt = (
        "You are an AI agricultural assistant. Provide accurate, concise, and helpful "
        "answers about farming practices, crop diseases, weather planning, and rural business questions."
    )

    try:
        response = gemini_model.generate_content(
            [prompt, user_message],
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "max_output_tokens": 512,
            },
        )
        ai_text = (response.text or "").strip()
        if not ai_text:
            ai_text = "I couldn't generate a response right now. Please try again."
        return jsonify({"status": "success", "response": ai_text})
    except Exception as exc:
        print(f"Error during agri chatbot response: {exc}")
        return jsonify({"status": "error", "error": "Failed to connect to Gemini API."}), 500



@app.route("/direct_sales_platform")
def direct_sales_platform_page():
    return render_template("direct_sales_platform.html")

@app.route("/particular_disease_detection")
def particular_disease_detection_page():
    return render_template("particular_disease_detection.html")

@app.route("/farmer_login", methods=["GET", "POST"])
def farmer_login_page():
    if request.method == "POST":
        farmer_name = request.form["farmer_name"]
        place = request.form["place"]
        mobile_number = request.form.get("mobile_number") # Use .get to handle optional fields
        email_id = request.form.get("email_id") # Use .get to handle optional fields
        
        db = get_db()
        db.execute("INSERT INTO farmers (name, place, mobile_number, email_id) VALUES (?, ?, ?, ?)",
                   (farmer_name, place, mobile_number, email_id))
        db.commit()
        db.close()

        # Redirect to farmer dashboard after successful login
        return redirect(url_for("farmer_dashboard"))
    return render_template("farmer_login.html")

@app.route("/farmer_dashboard")
def farmer_dashboard():
    return render_template("farmer_dashboard.html")

@app.route("/farmer_to_consumers", methods=["GET", "POST"])
def farmer_to_consumers():
    farmer_offers_df = pd.read_csv(FARMER_TO_CONSUMER_CSV)
    consumer_requests_df = pd.read_csv(CONSUMER_REQUEST_CSV)

    # Get unique CropNames from farmer offers for the dropdown
    crop_names = sorted(farmer_offers_df["CropName"].unique().tolist())

    matches = []
    selected_crop_name = request.form.get("CropName")

    if request.method == "POST" and selected_crop_name and selected_crop_name != "":
        print("--- POST Request Received ---")
        print(f"Raw form data: {request.form}")
        
        # Filter farmer offers by selected CropName
        filtered_farmer_offers = farmer_offers_df[farmer_offers_df["CropName"] == selected_crop_name]

        print(f"Selected CropName: {selected_crop_name}")
        print(f"Filtered farmer offers count: {len(filtered_farmer_offers)}")
        print(f"Filtered farmer offers head:\n{filtered_farmer_offers.head()}")

        if not filtered_farmer_offers.empty:
            for index, farmer_offer in filtered_farmer_offers.iterrows():
                farmer_crop_name = farmer_offer["CropName"]
                
                matching_consumer_requests = consumer_requests_df[
                    (consumer_requests_df["CropNeeded"].str.contains(farmer_crop_name, case=False, na=False))
                ]
                
                for req_idx, consumer_req in matching_consumer_requests.iterrows():
                    matches.append({
                        "ConsumerID": consumer_req["ConsumerID"],
                        "ConsumerName": consumer_req["ConsumerName"],
                        "MobileNumber": consumer_req["MobileNumber"],
                        "EmailID": consumer_req["EmailID"],
                        "CropNeeded": consumer_req["CropNeeded"],
                        "QuantityNeeded": consumer_req["QuantityNeeded"],
                        "MaxPrice": consumer_req["MaxPrice"],
                        "DeliveryPreference": consumer_req["DeliveryPreference"],
                        "Address": consumer_req["Address"],
                        "FarmerName": farmer_offer["FarmerName"],
                        "CropName": farmer_offer["CropName"],
                        "CropVariety": farmer_offer["CropVariety"],
                        "QuantityAvailable": farmer_offer["QuantityAvailable"],
                        "PricePerUnit": farmer_offer["PricePerUnit"],
                    })

    return render_template("farmer_to_consumers.html", crop_names=crop_names, matches=matches, selected_crop_name=selected_crop_name)

@app.route("/farmers_to_apartments", methods=["GET", "POST"])
def farmers_to_apartments():
    farmer_apartment_offers_df = pd.read_csv(FARMER_TO_APARTMENT_CSV)
    apartment_demands_df = pd.read_csv(APARTMENT_CROP_DEMAND_CSV)

    # Get unique CropNames from farmer offers for the dropdown
    crop_names = sorted(farmer_apartment_offers_df["CropName"].unique().tolist())

    matches = []
    selected_crop_name = request.form.get("CropName")

    if request.method == "POST" and selected_crop_name and selected_crop_name != "":
        # Filter farmer offers by selected CropName
        filtered_farmer_offers = farmer_apartment_offers_df[farmer_apartment_offers_df["CropName"] == selected_crop_name]

        if not filtered_farmer_offers.empty:
            for index, farmer_offer in filtered_farmer_offers.iterrows():
                # Find matching apartment demands based on CropWanted
                matching_apartment_demands = apartment_demands_df[
                    (apartment_demands_df["CropWanted"].str.contains(farmer_offer["CropName"], case=False, na=False))
                ]

                for req_idx, apartment_demand in matching_apartment_demands.iterrows():
                    matches.append({
                        "ApartmentID": apartment_demand["ApartmentID"],
                        "ApartmentName": apartment_demand["ApartmentName"],
                        "ApartmentAddress": apartment_demand["ApartmentAddress"],
                        "ContactPerson": apartment_demand["ContactPerson"],
                        "MobileNumber": apartment_demand["MobileNumber"],
                        "EmailID": apartment_demand["EmailID"],
                        "CropWanted": apartment_demand["CropWanted"],
                        "QuantityWanted": apartment_demand["QuantityWanted"],
                        "MaxPrice": apartment_demand["MaxPrice"],
                        "DeliveryPreference": apartment_demand["DeliveryPreference"],
                        # Add farmer offer details to the match as well for context
                        "FarmerName": farmer_offer["FarmerName"],
                        "FarmerCropName": farmer_offer["CropName"],
                        "FarmerCropVariety": farmer_offer["CropVariety"],
                        "FarmerQuantityAvailable": farmer_offer["QuantityAvailable"],
                        "FarmerPricePerUnit": farmer_offer["PricePerUnit"],
                    })

    return render_template("farmers_to_apartments.html", crop_names=crop_names, matches=matches, selected_crop_name=selected_crop_name)

@app.route("/consumer_login", methods=["GET", "POST"])
def consumer_login_page():
    if request.method == "POST":
        consumer_name = request.form["consumer_name"]
        phone_number = request.form["phone_number"]
        place = request.form["place"]
        email_id = request.form.get("email_id") # Use .get to handle optional fields

        db = get_db()
        db.execute("INSERT INTO consumers (name, phone_number, place, email_id) VALUES (?, ?, ?, ?)",
                   (consumer_name, phone_number, place, email_id))
        db.commit()
        db.close()

        return redirect(url_for("consumer_dashboard"))
    return render_template("consumer_login.html")

@app.route("/consumer_dashboard")
def consumer_dashboard():
    return render_template("consumer_dashboard.html")


@app.route("/consumer_to_farmers", methods=["GET", "POST"])
def consumer_to_farmers():
    consumer_crop_requests_df = pd.read_csv(CONSUMER_CROP_REQUEST_DATASET_CSV)
    farmer_data_df = pd.read_csv(CONSUMER_FARMER_DATASET_CSV)

    # Get unique CropNeeded from consumer requests for the dropdown
    crop_needed_options = sorted(consumer_crop_requests_df["CropNeeded"].unique().tolist())

    matches = []
    selected_crop_needed = request.form.get("CropNeeded")

    if request.method == "POST" and selected_crop_needed and selected_crop_needed != "":
        # Filter consumer requests by selected CropNeeded
        filtered_consumer_requests = consumer_crop_requests_df[consumer_crop_requests_df["CropNeeded"] == selected_crop_needed]

        if not filtered_consumer_requests.empty:
            for index, consumer_req in filtered_consumer_requests.iterrows():
                # Find matching farmer offers based on CropName
                matching_farmer_offers = farmer_data_df[
                    (farmer_data_df["CropName"].str.contains(consumer_req["CropNeeded"], case=False, na=False))
                ]
                for offer_idx, farmer_offer in matching_farmer_offers.iterrows():
                    matches.append({
                        "ConsumerID": consumer_req["ConsumerID"],
                        "ConsumerName": consumer_req["ConsumerName"],
                        "ConsumerMobileNumber": consumer_req["MobileNumber"],
                        "ConsumerEmailID": consumer_req["EmailID"],
                        "ConsumerAddress": consumer_req["Address"],
                        "CropNeeded": consumer_req["CropNeeded"],
                        "QuantityNeeded": consumer_req["QuantityNeeded"],
                        "MaxPrice": consumer_req["MaxPrice"],
                        "DeliveryPreference": consumer_req["DeliveryPreference"],
                        
                        "FarmerID": farmer_offer["FarmerID"],
                        "FarmerName": farmer_offer["FarmerName"],
                        "FarmerMobileNumber": farmer_offer["MobileNumber"],
                        "FarmerEmailID": farmer_offer["EmailID"],
                        "FarmerFarmAddress": farmer_offer["FarmAddress"],
                        "FarmerCropName": farmer_offer["CropName"],
                        "FarmerCropVariety": farmer_offer["CropVariety"],
                        "FarmerQuantityAvailable": farmer_offer["QuantityAvailable"],
                        "FarmerPricePerUnit": farmer_offer["PricePerUnit"],
                        "FarmerCropQuality": farmer_offer["CropQuality"],
                        "FarmerCropHealthCheck": farmer_offer["CropHealthCheck"],
                        "FarmerDeliveryRangeKM": farmer_offer["DeliveryRangeKM"],
                    })

    return render_template("consumer_to_farmers.html", crop_needed_options=crop_needed_options, matches=matches, selected_crop_needed=selected_crop_needed)


@app.route("/lemon_disease", methods=["GET", "POST"])
def lemon_disease_page():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "lemon")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "lemon", filename)
                prediction = predict_lemon_disease(filepath)
            except Exception as e:
                print(f"Error during lemon disease prediction: {e}")
                error = str(e) if "efficientnet_pytorch" in str(e) else "Failed to analyze the image. Please try again with a valid image."

    return render_template(
        "lemon_disease.html",
        prediction=prediction,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/potato_disease", methods=["GET", "POST"])
def potato_disease_page():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "potato")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "potato", filename)
                prediction = predict_potato_disease(filepath)
            except Exception as e:
                print(f"Error during potato disease prediction: {e}")
                error = "Failed to analyze the image. Please try again with a valid image."

    return render_template(
        "potato_disease.html",
        prediction=prediction,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/rice_disease", methods=["GET", "POST"])
def rice_disease_page():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "rice")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "rice", filename)
                prediction = predict_rice_disease(filepath)
            except Exception as e:
                print(f"Error during rice disease prediction: {e}")
                error = "Failed to analyze the image. Please try again with a valid image."

    return render_template(
        "rice_disease.html",
        prediction=prediction,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/tomato_disease", methods=["GET", "POST"])
def tomato_disease_page():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "tomato")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "tomato", filename)
                prediction = predict_tomato_disease(filepath)
            except Exception as e:
                print(f"Error during tomato disease prediction: {e}")
                error = "Failed to analyze the image. Please try again with a valid image."

    return render_template(
        "tomato_disease.html",
        prediction=prediction,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/cucumber_disease", methods=["GET", "POST"])
def cucumber_disease_page():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "cucumber")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_filename = os.path.join("uploads", "cucumber", filename)
                prediction = predict_cucumber_disease(filepath)
            except Exception as e:
                print(f"Error during cucumber disease prediction: {e}")
                error = "Failed to analyze the image. Please try again with a valid image."

    return render_template(
        "cucumber_disease.html",
        prediction=prediction,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/karnataka_yield_prediction", methods=["GET", "POST"])
def karnataka_yield_prediction_page():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            form_data = {
                "District": request.form["district"],
                "crop": request.form["crop"],
                "croptype": request.form["croptype"],
                "Area": request.form["area"],
                "Production": request.form["production"],
            }
            predicted_yield = predict_karnataka_yield(form_data)
            prediction = {
                "district": form_data["District"],
                "crop": form_data["crop"],
                "croptype": form_data["croptype"],
                "area": float(form_data["Area"]),
                "production": float(form_data["Production"]),
                "yield_value": predicted_yield,
            }
        except ValueError as e:
            error = str(e)
        except Exception as e:
            print(f"Error during Karnataka yield prediction: {e}")
            error = "An unexpected error occurred during prediction."

    return render_template(
        "karnataka_yield_prediction.html",
        prediction=prediction,
        error=error,
        karnataka_categorical_options=karnataka_categorical_options,
    )

# E-Learning Video Data
# This is a placeholder, you would populate this with actual video details
ELE_VIDEOS = {
    "by_product_making": [
        {
            "title": "How to Make Ragi Malt at Home",
            "url": "https://www.youtube.com/results?search_query=%E0%B2%B0%E0%B2%BE%E0%B2%97%E0%B2%BF+%E0%B2%AE%E0%B2%BE%E0%B2%B2%E0%B3%8D%E0%B2%9F%E0%B3%8D+%E0%B2%AE%E0%B2%BE%E0%B2%A1%E0%B3%81%E0%B2%B5+%E0%B2%B5%E0%B2%BF%E0%B2%A7%E0%B2%BE%E0%B2%A8",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Krushi Jagattu (Kannada Agriculture Channel)",
            "url": "https://www.youtube.com/@KrushiJagattu",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Mushroom Powder Processing (Kannada)",
            "url": "https://www.youtube.com/results?search_query=%E0%B2%AE%E0%B2%B6%E0%B3%8D%E0%B2%B0%E0%B3%82%E0%B2%AE%E0%B3%8D+%E0%B2%AA%E0%B3%8C%E0%B2%A1%E0%B2%B0%E0%B3%8D+%E0%B2%AE%E0%B2%BE%E0%B2%A1%E0%B3%81%E0%B2%B5+%E0%B2%B5%E0%B2%BF%E0%B2%A7%E0%B2%BE%E0%B2%A8",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Udayavani Krushi YouTube",
            "url": "https://www.youtube.com/@UdayavaniAgri",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
    ],
    "documentation_registration": [
        {
            "title": "FSSAI Registration Explained in Kannada",
            "url": "https://www.youtube.com/results?search_query=FSSAI+registration+Kannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Legal Helpline Kannada",
            "url": "https://www.youtube.com/@LegalHelplineKannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "GST Registration Kannada Explanation",
            "url": "https://www.youtube.com/results?search_query=GST+registration+Kannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Taxation Kannada Channel",
            "url": "https://www.youtube.com/@KannadaTax",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
    ],
    "startup_marketing_guidance": [
        {
            "title": "Marketing Tips for Farmers (Kannada)",
            "url": "https://www.youtube.com/results?search_query=%E0%B2%B0%E0%B3%88%E0%B2%A4%E0%B2%B0%E0%B2%BF%E0%B2%97%E0%B3%86+%E0%B2%AE%E0%B2%BE%E0%B2%B0%E0%B3%8D%E0%B2%95%E0%B3%86%E0%B2%9F%E0%B2%BF%E0%B2%82%E0%B2%97%E0%B3%8D+%E0%B2%A4%E0%B2%82%E0%B2%A4%E0%B3%8D%E0%B2%B0%E0%B2%97%E0%B2%B3%E0%B3%81",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "TV9 Krushi Special",
            "url": "https://www.youtube.com/@TV9Kannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Branding & Packaging for Food Products (Kannada)",
            "url": "https://www.youtube.com/results?search_query=%E0%B2%AB%E0%B3%81%E0%B2%A1%E0%B3%8D+%E0%B2%AA%E0%B3%8D%E0%B2%AF%E0%B2%BE%E0%B2%95%E0%B3%87%E0%B2%9C%E0%B2%BF%E0%B2%82%E0%B2%97%E0%B3%8D+%E0%B2%95%E0%B2%A8%E0%B3%8D%E0%B2%A8%E0%B2%A1",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "Public TV Krushi",
            "url": "https://www.youtube.com/@PublicTVKannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
    ],
    "govt_schemes_loans": [
        {
            "title": "PMFME Scheme Explained in Kannada",
            "url": "https://www.youtube.com/results?search_query=PMFME+scheme+Kannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
        {
            "title": "KCC Loan Kannada Explanation",
            "url": "https://www.youtube.com/results?search_query=Kisan+credit+card+Kannada",
            "thumbnail": None,
            "duration": "",
            "language": "Kannada | Free | Farmer Friendly",
        },
    ],
}

@app.route("/elearning")
def elearning_page():
    return render_template("elearning.html", videos=ELE_VIDEOS)


def get_lemon_model():
    global lemon_model
    if lemon_model is None:
        model = models.mobilenet_v2(pretrained=False)  # Changed to MobileNetV2
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(LEMON_CLASS_NAMES))
        state_dict = torch.load(LEMON_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        lemon_model = model
    return lemon_model


def predict_lemon_disease(image_path, confidence_threshold=0.80):
    model = get_lemon_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = lemon_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
    
    confidence = score.item()
    predicted_class = LEMON_CLASS_NAMES[pred_idx.item()]

    if confidence < confidence_threshold:
        result_message = "Unknown class"
        is_diseased = False
        is_healthy = False
    else:
        is_healthy = "healthy" in predicted_class.lower()
        is_diseased = not is_healthy
        if is_diseased:
            result_message = "It is diseased"
        elif is_healthy:
            result_message = "It is healthy"
        else:
            result_message = predicted_class # Fallback if neither diseased nor healthy is explicit

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "is_diseased": is_diseased,
        "result_message": result_message,
    }


def get_potato_model():
    global potato_model
    if potato_model is None:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(POTATO_CLASS_NAMES))
        state_dict = torch.load(POTATO_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        potato_model = model
    return potato_model


def predict_potato_disease(image_path, confidence_threshold=0.80):
    model = get_potato_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = potato_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
    
    confidence = score.item()
    predicted_class = POTATO_CLASS_NAMES[pred_idx.item()]

    if confidence < confidence_threshold:
        result_message = "Unknown class"
        is_diseased = False
        is_healthy = False
    else:
        is_healthy = "healthy" in predicted_class.lower()
        is_diseased = not is_healthy
        if is_diseased:
            result_message = "It is diseased"
        elif is_healthy:
            result_message = "It is healthy"
        else:
            result_message = predicted_class # Fallback if neither diseased nor healthy is explicit

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "is_diseased": is_diseased,
        "result_message": result_message,
    }


def get_rice_model():
    global rice_model
    if rice_model is None:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(RICE_CLASS_NAMES))
        state_dict = torch.load(RICE_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        rice_model = model
    return rice_model


def predict_rice_disease(image_path, confidence_threshold=0.80):
    model = get_rice_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = rice_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
    
    confidence = score.item()
    predicted_class = RICE_CLASS_NAMES[pred_idx.item()]

    if confidence < confidence_threshold:
        result_message = "Unknown class"
        is_diseased = False
        is_healthy = False
    else:
        is_healthy = "healthy" in predicted_class.lower()
        is_diseased = not is_healthy
        if is_diseased:
            result_message = "It is diseased"
        elif is_healthy:
            result_message = "It is healthy"
        else:
            result_message = predicted_class # Fallback if neither diseased nor healthy is explicit

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "is_diseased": is_diseased,
        "result_message": result_message,
    }


def get_tomato_model():
    global tomato_model
    if tomato_model is None:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(TOMATO_CLASS_NAMES))
        state_dict = torch.load(TOMATO_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        tomato_model = model
    return tomato_model


def predict_tomato_disease(image_path, confidence_threshold=0.80):
    model = get_tomato_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = tomato_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
    
    confidence = score.item()
    predicted_class = TOMATO_CLASS_NAMES[pred_idx.item()]

    if confidence < confidence_threshold:
        result_message = "Unknown class"
        is_diseased = False
        is_healthy = False
    else:
        is_healthy = "healthy" in predicted_class.lower()
        is_diseased = not is_healthy
        if is_diseased:
            result_message = "It is diseased"
        elif is_healthy:
            result_message = "It is healthy"
        else:
            result_message = predicted_class # Fallback if neither diseased nor healthy is explicit

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "is_diseased": is_diseased,
        "result_message": result_message,
    }


def get_cucumber_model():
    global cucumber_model
    if cucumber_model is None:
        model = models.resnet50(pretrained=False) # Changed to ResNet50
        model.fc = torch.nn.Linear(model.fc.in_features, len(CUCUMBER_CLASS_NAMES))
        state_dict = torch.load(CUCUMBER_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        cucumber_model = model
    return cucumber_model


def predict_cucumber_disease(image_path, confidence_threshold=0.80):
    model = get_cucumber_model()
    image = Image.open(image_path).convert("RGB")
    img_tensor = cucumber_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)
    
    confidence = score.item()
    predicted_class = CUCUMBER_CLASS_NAMES[pred_idx.item()]

    if confidence < confidence_threshold:
        result_message = "Unknown class"
        is_diseased = False
        is_healthy = False
    else:
        is_healthy = "healthy" in predicted_class.lower()
        is_diseased = not is_healthy
        if is_diseased:
            result_message = "It is diseased"
        elif is_healthy:
            result_message = "It is healthy"
        else:
            result_message = predicted_class # Fallback if neither diseased nor healthy is explicit

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "is_diseased": is_diseased,
        "result_message": result_message,
    }


def preprocess_karnataka_features(form_data):
    encoded = []
    for col in ["District", "crop", "croptype"]:
        encoder = karnataka_label_encoders[col]
        try:
            encoded_value = encoder.transform([form_data[col]])[0]
        except ValueError:
            raise ValueError(f"Unseen label for {col}: '{form_data[col]}'. Please choose another value.")
        encoded.append(encoded_value)

    try:
        area = float(form_data["Area"])
        production = float(form_data["Production"])
    except ValueError:
        raise ValueError("Area and Production must be numeric.")

    if area <= 0:
        raise ValueError("Area must be greater than zero.")
    if production < 0:
        raise ValueError("Production cannot be negative.")

    encoded.extend([area, production])
    return np.array([encoded])


def predict_karnataka_yield(form_data):
    features = preprocess_karnataka_features(form_data)
    prediction = karnataka_yield_model.predict(features)[0]
    return prediction


# --------------------------

# Run the app

# --------------------------

if __name__ == "__main__":

    print("✅ Running from:", BASE_DIR)
    init_db()

    # Path for storing uploaded images (for disease detection)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    # New folder for pre-stored disease images
    PRESET_DISEASE_IMAGES_FOLDER = os.path.join(BASE_DIR, "static", "preset_disease_images")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # Create UPLOAD_FOLDER and PRESET_DISEASE_IMAGES_FOLDER if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PRESET_DISEASE_IMAGES_FOLDER, exist_ok=True)

    def open_browser():
        url = "http://127.0.0.1:5000/"
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        try:
            if os.path.exists(chrome_path):
                webbrowser.register(
                    "chrome",
                    None,
                    webbrowser.BackgroundBrowser(chrome_path),
                )
                webbrowser.get("chrome").open(url)
            else:
                webbrowser.open(url)
        except webbrowser.Error as e:
            print(f"Unable to open browser automatically: {e}")

    threading.Timer(1.0, open_browser).start()

    app.run(debug=True)
