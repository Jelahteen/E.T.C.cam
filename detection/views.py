from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.translation import get_language
from django.core.files.storage import default_storage
from django.conf import settings

import os
import io
import random

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from .models import Plant, Disease, Scan
from django.contrib.auth import get_user_model

User = get_user_model()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# Set to False in production when you want to use real models
TEST_MODE = True
print(f"TEST_MODE: {TEST_MODE}")

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(os.path.join(settings.BASE_DIR, TEMP_UPLOAD_DIR), exist_ok=True)

# Paths to .pth models
CHILI_MODEL_PATH = os.path.join(settings.BASE_DIR, "detection", "models", "chili_resnet.pth")
EGGPLANT_MODEL_PATH = os.path.join(settings.BASE_DIR, "detection", "models", "eggplant_resnet.pth")
TOMATO_LEAF_MODEL_PATH = os.path.join(settings.BASE_DIR, "detection", "models", "tomato_leaf_resnet.pth")

# PyTorch model objects
chili_model = None
eggplant_model = None
tomato_leaf_model = None

loaded_models = {
    "chili": False,
    "eggplant": False,
    "tomato_leaf": False,
}

# Test-mode canned predictions (unchanged)
TEST_PREDICTIONS = {
    "eggplant": [
        (
            "Fresh Eggplant",
            0.95,
            "The eggplant appears healthy and fresh.",
            "No signs of disease or damage.",
            "Continue current care practices.",
        ),
        (
            "Powdery Mildew",
            0.87,
            "Fungal disease causing white powdery spots.",
            "White powdery growth on leaves and stems.",
            "Apply fungicide and improve air circulation.",
        ),
        (
            "Leaf Wilt",
            0.78,
            "Plant showing signs of water stress or disease.",
            "Drooping or wilting leaves, possible discoloration.",
            "Check watering schedule and soil drainage.",
        ),
    ],
    "chili": [
        (
            "Ripe",
            0.92,
            "Chili peppers are fully ripe and ready for harvest.",
            "Bright red color, firm texture.",
            "Harvest promptly for best flavor.",
        ),
        (
            "Unripe",
            0.85,
            "Chili peppers are still developing.",
            "Green color, firm but not fully sized.",
            "Allow more time to mature.",
        ),
        (
            "Bacterial Spot",
            0.76,
            "Bacterial infection affecting leaves and fruit.",
            "Dark spots with yellow halos on leaves.",
            "Use copper-based bactericides.",
        ),
    ],
    "tomato": [
        (
            "Healthy Tomato Leaves",
            0.94,
            "Tomato plant appears healthy and vigorous.",
            "Green leaves, strong stem growth.",
            "Maintain current care and monitoring.",
        ),
        (
            "Early Blight",
            0.82,
            "Fungal disease causing concentric ring spots.",
            "Brown spots with concentric rings on lower leaves.",
            "Remove affected leaves and apply fungicide.",
        ),
        (
            "Late Blight",
            0.79,
            "Serious fungal disease that can destroy plants.",
            "Water-soaked spots that turn brown and papery.",
            "Apply fungicide immediately and improve air flow.",
        ),
    ],
}

# ----------------------------------------------------------------------
# MODEL LOADING (ONLY .PTH)
# ----------------------------------------------------------------------

if not TEST_MODE:
    try:
        if os.path.exists(CHILI_MODEL_PATH):
            try:
                chili_model = torch.load(
                    CHILI_MODEL_PATH,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                chili_model.eval()
                loaded_models["chili"] = True
            except Exception as e:
                print(f"Failed to load chili model: {e}")

        if os.path.exists(EGGPLANT_MODEL_PATH):
            try:
                eggplant_model = torch.load(
                    EGGPLANT_MODEL_PATH,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                eggplant_model.eval()
                loaded_models["eggplant"] = True
            except Exception as e:
                print(f"Failed to load eggplant model: {e}")

        if os.path.exists(TOMATO_LEAF_MODEL_PATH):
            try:
                tomato_leaf_model = torch.load(
                    TOMATO_LEAF_MODEL_PATH,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                tomato_leaf_model.eval()
                loaded_models["tomato_leaf"] = True
            except Exception as e:
                print(f"Failed to load tomato leaf model: {e}")

    except Exception as e:
        print(f"Model loading error: {e}")

models_loaded_successfully = any(loaded_models.values())

# ----------------------------------------------------------------------
# IMAGE PREPROCESSING & PREDICTION (PYTORCH ONLY)
# ----------------------------------------------------------------------


def preprocess_image_torch(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"PyTorch preprocessing error: {e}")
        return None


def predict_pytorch(model, image_tensor, class_names):
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(probabilities, 0)
            predicted_label = class_names[predicted_idx.item()]
            confidence = probabilities[predicted_idx.item()].item()
        return predicted_label, confidence
    except Exception as e:
        print(f"PyTorch prediction error: {e}")
        return None, None


# ----------------------------------------------------------------------
# VIEWS
# ----------------------------------------------------------------------


@login_required
def scan_view(request):
    return render(request, "detection/Scanpage.html")


def debug_database(request):
    plants = Plant.objects.all()
    result = {}
    for plant in plants:
        diseases = Disease.objects.filter(plants=plant)
        result[plant.name] = [
            {
                "name": disease.name,
                "name_fil": disease.get_translated_name("tl"),
                "description": disease.description[:50] + "..."
                if disease.description
                else "None",
                "description_fil": disease.get_translated_description("tl")[
                    :50
                ]
                + "..."
                if disease.get_translated_description("tl")
                else "None",
            }
            for disease in diseases
        ]
    return JsonResponse(result)


def disease_list_view(request):
    plants_with_diseases = []
    current_language = get_language()

    for plant in Plant.objects.all().prefetch_related("diseases"):
        diseases = plant.diseases.all()
        plant_name = (
            plant.get_translated_name("tl")
            if current_language == "tl"
            else plant.name
        )
        plant_data = {
            "name": plant_name,
            "original_name": plant.name,
            "diseases": [],
        }

        for disease in diseases:
            if current_language == "tl":
                disease_data = {
                    "id": disease.id,
                    "name": disease.get_translated_name("tl"),
                    "description": disease.get_translated_description("tl"),
                    "symptoms": disease.get_translated_symptoms("tl"),
                    "prevention": disease.get_translated_prevention("tl"),
                    "original_name": disease.name,
                    "translation_status": disease.translation_status(),
                }
            else:
                disease_data = {
                    "id": disease.id,
                    "name": disease.name,
                    "description": disease.description,
                    "symptoms": disease.symptoms,
                    "prevention": disease.prevention,
                    "original_name": disease.name,
                    "translation_status": disease.translation_status(),
                }
            plant_data["diseases"].append(disease_data)
        plants_with_diseases.append(plant_data)

    context = {
        "plants_with_diseases": plants_with_diseases,
        "current_language": current_language,
    }
    return render(request, "detection/list_page.html", context)


def get_current_language(request):
    current_language = get_language()
    post_language = request.POST.get("language", "")
    if post_language:
        current_language = post_language
    elif not current_language:
        accept_language = request.META.get("HTTP_ACCEPT_LANGUAGE", "")
        if accept_language:
            primary_lang = accept_language.split(",")[0].split("-")[0].lower()
            if primary_lang in ["en", "tl"]:
                current_language = primary_lang
    if not current_language:
        current_language = "en"
    return current_language


# ----------------------------------------------------------------------
# TEST MODE HANDLER (UNCHANGED LOGIC)
# ----------------------------------------------------------------------


def handle_test_mode(request, plant_type_str, uploaded_image):
    current_language = get_current_language(request)

    available_predictions = TEST_PREDICTIONS.get(plant_type_str, [])
    if available_predictions:
        (
            prediction_label,
            confidence,
            description,
            symptoms,
            prevention,
        ) = random.choice(available_predictions)
    else:
        prediction_label = "No test data available"
        confidence = 0.0
        description = (
            "Test mode is active but no test predictions configured for this plant type."
        )
        symptoms = "N/A"
        prevention = "N/A"

    plant_mapping = {
        "chili": "Chili Pepper",
        "eggplant": "Eggplant",
        "tomato": "Tomato",
    }
    db_plant_name = plant_mapping.get(plant_type_str)
    plant_obj = None
    disease_obj = None

    if db_plant_name:
        try:
            plant_obj = Plant.objects.get(name=db_plant_name)
            try:
                disease_obj = Disease.objects.get(
                    name__iexact=prediction_label, plants=plant_obj
                )
            except Exception:
                pass
        except Exception:
            pass

    final_disease_name = prediction_label
    final_description = description
    final_symptoms = symptoms
    final_prevention = prevention

    if current_language == "tl" and disease_obj:
        final_disease_name = (
            disease_obj.get_translated_name("tl") or prediction_label
        )
        final_description = (
            disease_obj.get_translated_description("tl") or description
        )
        final_symptoms = (
            disease_obj.get_translated_symptoms("tl") or symptoms
        )
        final_prevention = (
            disease_obj.get_translated_prevention("tl") or prevention
        )

    if plant_obj:
        try:
            Scan.objects.create(
                user=request.user,
                plant=plant_obj,
                disease=disease_obj,
                scanned_image=uploaded_image,
                detected_result=prediction_label,
                confidence=confidence if confidence else 0.0,
            )
        except Exception:
            pass

    return JsonResponse(
        {
            "disease": final_disease_name,
            "confidence": f"{confidence*100:.2f}%" if confidence else "N/A",
            "description": final_description,
            "symptoms": final_symptoms,
            "prevention": final_prevention,
            "test_mode": True,
            "language": current_language,
            "note": "This result is from test mode - using simulated predictions",
        }
    )


# ----------------------------------------------------------------------
# REAL PREDICTIONS (.PTH ONLY)
# ----------------------------------------------------------------------


def make_real_prediction(plant_type_str, full_file_path):
    prediction_label = "No disease detected"
    confidence = 0.0

    if plant_type_str == "eggplant" and eggplant_model:
        eggplant_class_names = [
            "Aphids",
            "Cercospora Leaf Spot",
            "Defect Eggplant",
            "Flea Beetles",
            "Fresh Eggplant",
            "Fresh Eggplant Leaf",
            "Leaf Wilt",
            "Phytophthora Blight",
            "Powdery Mildew",
            "Tobacco Mosaic Virus",
        ]
        processed_image = preprocess_image_torch(full_file_path)
        if processed_image is not None:
            prediction_label, confidence = predict_pytorch(
                eggplant_model, processed_image, eggplant_class_names
            )

    elif plant_type_str == "chili" and chili_model:
        chili_class_names = [
            "Damaged",
            "Dried",
            "Old",
            "Ripe",
            "Unripe",
        ]
        processed_image = preprocess_image_torch(full_file_path)
        if processed_image is not None:
            prediction_label, confidence = predict_pytorch(
                chili_model, processed_image, chili_class_names
            )

    elif plant_type_str == "tomato" and tomato_leaf_model:
        tomato_leaf_class_names = [
            "Bacterial Spot",
            "Early Blight",
            "Healthy Tomato Leaves",
            "Late Blight",
            "Leaf Mold",
            "Powdery Mildew",
            "Septoria Leaf Spot",
            "Spider Mites",
            "Target Spot",
            "Tomato Mosaic Virus",
            "Tomato Yellow Leaf Curl Virus",
        ]
        processed_image = preprocess_image_torch(full_file_path)
        if processed_image is not None:
            leaf_prediction, leaf_conf = predict_pytorch(
                tomato_leaf_model, processed_image, tomato_leaf_class_names
            )

            prediction_label = leaf_prediction or "No disease detected"
            confidence = leaf_conf or 0.0

    return prediction_label or "No disease detected", confidence or 0.0


def get_disease_info(
    prediction_label, plant_type_str, current_language
):
    plant_mapping = {
        "chili": "Chili Pepper",
        "eggplant": "Eggplant",
        "tomato": "Tomato",
    }
    db_plant_name = plant_mapping.get(plant_type_str)
    plant_obj = None
    disease_obj = None

    if db_plant_name:
        try:
            plant_obj = Plant.objects.get(name=db_plant_name)
            disease_obj = (
                Disease.objects.filter(
                    name__iexact=prediction_label, plants=plant_obj
                ).first()
            )
        except Exception:
            pass

    non_disease_labels = {
        "Damaged",
        "Dried",
        "Old",
        "Ripe",
        "Unripe",
        "Defect Eggplant",
        "Fresh Eggplant",
        "Fresh Eggplant Leaf",
        "Healthy Tomato Leaves",
        "Healthy",
        "No disease detected",
        "Chili Pepper",
        "Eggplant",
        "Tomato",
    }

    is_disease = prediction_label not in non_disease_labels

    disease_description = "No description available."
    disease_symptoms = "No symptoms information available."
    disease_prevention = "No prevention strategies available."

    if disease_obj:
        if current_language == "tl":
            final_disease_name = (
                disease_obj.get_translated_name("tl") or disease_obj.name
            )
            disease_description = (
                disease_obj.get_translated_description("tl")
                or disease_obj.description
                or disease_description
            )
            disease_symptoms = (
                disease_obj.get_translated_symptoms("tl")
                or disease_obj.symptoms
                or disease_symptoms
            )
            disease_prevention = (
                disease_obj.get_translated_prevention("tl")
                or disease_obj.prevention
                or disease_prevention
            )
        else:
            final_disease_name = disease_obj.name
            disease_description = (
                disease_obj.description or disease_description
            )
            disease_symptoms = disease_obj.symptoms or disease_symptoms
            disease_prevention = (
                disease_obj.prevention or disease_prevention
            )
    else:
        final_disease_name = prediction_label
        if not is_disease:
            if prediction_label in ["Healthy Tomato Leaves", "Healthy"]:
                disease_description = (
                    "The plant appears healthy based on the scan."
                )
                disease_symptoms = "No symptoms of disease detected."
                disease_prevention = (
                    "Continue good agricultural practices."
                )
            elif prediction_label == "No disease detected":
                disease_description = (
                    "The scan did not detect any specific disease."
                )
                disease_symptoms = "No specific symptoms identified."
                disease_prevention = (
                    "Monitor the plant for any changes."
                )
            else:
                disease_description = (
                    f"The scan identified the plant as: {prediction_label}. "
                    "This is a general condition, not a disease."
                )
                disease_symptoms = (
                    "This condition indicates the plant's growth stage or "
                    "quality, not disease symptoms."
                )
                disease_prevention = (
                    "No specific disease prevention needed. "
                    "Focus on general plant care practices."
                )

    return (
        final_disease_name,
        disease_description,
        disease_symptoms,
        disease_prevention,
        plant_obj,
        disease_obj if is_disease else None,
        is_disease,
    )


# ----------------------------------------------------------------------
# DETECT VIEW
# ----------------------------------------------------------------------


@login_required
@csrf_exempt
def detect_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        plant_type_str = request.POST.get("plant_type", "").lower()
        uploaded_image = request.FILES.get("image")

        if not uploaded_image or not plant_type_str:
            return JsonResponse(
                {"error": "Missing image or plant type"}, status=400
            )

        # TEST MODE SHORT-CIRCUIT
        if TEST_MODE:
            return handle_test_mode(
                request, plant_type_str, uploaded_image
            )

        # Ensure at least one model loaded
        if not models_loaded_successfully:
            return JsonResponse(
                {"error": "No machine learning models are available"},
                status=503,
            )

        plant_model_available = (
            (plant_type_str == "eggplant" and loaded_models["eggplant"])
            or (plant_type_str == "chili" and loaded_models["chili"])
            or (plant_type_str == "tomato" and loaded_models["tomato_leaf"])
        )
        if not plant_model_available:
            return JsonResponse(
                {"error": f"Model for {plant_type_str} is not available"},
                status=503,
            )

        # Save temp file
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_image.name)
        saved_file_name = default_storage.save(
            temp_file_path, uploaded_image
        )
        full_file_path = default_storage.path(saved_file_name)

        # Run prediction
        prediction_label, confidence = make_real_prediction(
            plant_type_str, full_file_path
        )

        # Delete temp file
        if "saved_file_name" in locals() and default_storage.exists(
            saved_file_name
        ):
            default_storage.delete(saved_file_name)

        # Build response from DB + translation
        current_language = get_current_language(request)
        (
            final_disease_name,
            disease_description,
            disease_symptoms,
            disease_prevention,
            plant_obj,
            disease_obj,
            is_disease,
        ) = get_disease_info(
            prediction_label, plant_type_str, current_language
        )

        # Save scan record
        if plant_obj:
            try:
                Scan.objects.create(
                    user=request.user,
                    plant=plant_obj,
                    disease=disease_obj,
                    scanned_image=uploaded_image,
                    detected_result=prediction_label,
                    confidence=confidence if confidence else 0.0,
                )
            except Exception as e:
                print(f"Error saving Scan: {e}")

        return JsonResponse(
            {
                "disease": final_disease_name,
                "confidence": f"{confidence*100:.2f}%"
                if confidence
                else "N/A",
                "description": disease_description,
                "symptoms": disease_symptoms,
                "prevention": disease_prevention,
                "language": current_language,
            }
        )

    except Exception as e:
        print(f"Error in detect_view: {str(e)}")
        if "saved_file_name" in locals() and default_storage.exists(
            saved_file_name
        ):
            default_storage.delete(saved_file_name)
        return JsonResponse(
            {"error": f"Server error: {str(e)}"}, status=500
        )
        return JsonResponse({'error': 'Invalid request method'}, status=405)