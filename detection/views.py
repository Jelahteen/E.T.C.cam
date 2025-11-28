from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.translation import get_language, activate
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
from torchvision import transforms
from django.core.files.storage import default_storage
from django.conf import settings
import json
import logging

# Import your models from the detection app and the User model
from .models import Plant, Disease, Scan
from django.contrib.auth import get_user_model

User = get_user_model()

# =============================================================================
# TEST MODE SETTINGS - SET TO True FOR TESTING WITHOUT REAL MODELS
# =============================================================================
TEST_MODE = True  # Set to False when you have real model files
print(f"🔧 DEBUG: TEST_MODE is set to: {TEST_MODE}")
# =============================================================================

# Ensure 'temp_uploads' directory exists for temporary image storage
TEMP_UPLOAD_DIR = 'temp_uploads'
if not os.path.exists(os.path.join(settings.BASE_DIR, TEMP_UPLOAD_DIR)):
    os.makedirs(os.path.join(settings.BASE_DIR, TEMP_UPLOAD_DIR))

# CORRECTED: Update model paths to point to detection/models
CHILI_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'chili_resnet.pth')
EGGPLANT_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'eggplant_resnet.pth')
TOMATO_LEAF_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'tomato_leaf_resnet.pth')
TOMATO_VEGE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'tomato_vege_model.keras')

print(f"🔧 DEBUG: Model paths:")
print(f"  - Chili: {CHILI_MODEL_PATH}")
print(f"  - Eggplant: {EGGPLANT_MODEL_PATH}")
print(f"  - Tomato Leaf: {TOMATO_LEAF_MODEL_PATH}")
print(f"  - Tomato Vege: {TOMATO_VEGE_MODEL_PATH}")

# Initialize models as None
chili_model = None
eggplant_model = None
tomato_leaf_model = None
tomato_vege_model = None

# Track which models loaded successfully
loaded_models = {
    'chili': False,
    'eggplant': False, 
    'tomato_leaf': False,
    'tomato_vege': False
}

# Test mode predictions - these will be used when TEST_MODE = True
TEST_PREDICTIONS = {
    'eggplant': [
        ('Fresh Eggplant', 0.95, "The eggplant appears healthy and fresh.", "No signs of disease or damage.", "Continue current care practices."),
        ('Powdery Mildew', 0.87, "Fungal disease causing white powdery spots.", "White powdery growth on leaves and stems.", "Apply fungicide and improve air circulation."),
        ('Leaf Wilt', 0.78, "Plant showing signs of water stress or disease.", "Drooping or wilting leaves, possible discoloration.", "Check watering schedule and soil drainage.")
    ],
    'chili': [
        ('Ripe', 0.92, "Chili peppers are fully ripe and ready for harvest.", "Bright red color, firm texture.", "Harvest promptly for best flavor."),
        ('Unripe', 0.85, "Chili peppers are still developing.", "Green color, firm but not fully sized.", "Allow more time to mature."),
        ('Bacterial Spot', 0.76, "Bacterial infection affecting leaves and fruit.", "Dark spots with yellow halos on leaves.", "Use copper-based bactericides.")
    ],
    'tomato': [
        ('Healthy Tomato Leaves', 0.94, "Tomato plant appears healthy and vigorous.", "Green leaves, strong stem growth.", "Maintain current care and monitoring."),
        ('Early Blight', 0.82, "Fungal disease causing concentric ring spots.", "Brown spots with concentric rings on lower leaves.", "Remove affected leaves and apply fungicide."),
        ('Late Blight', 0.79, "Serious fungal disease that can destroy plants.", "Water-soaked spots that turn brown and papery.", "Apply fungicide immediately and improve air flow.")
    ]
}

if not TEST_MODE:
    # REAL MODEL LOADING - Only execute if not in test mode
    print("🔧 DEBUG: Entering REAL model loading block")
    try:
        print("🔧 Attempting to load REAL machine learning models...")
        models_dir = os.path.join(settings.BASE_DIR, 'detection', 'models')
        
        print(f"🔧 DEBUG: Looking for models in: {models_dir}")
        print(f"🔧 DEBUG: Directory exists: {os.path.exists(models_dir)}")
        
        if not os.path.exists(models_dir):
            print(f"❌ ERROR: 'models' directory not found at {models_dir}")
        else:
            print(f"✅ Found models directory: {models_dir}")
            available_files = os.listdir(models_dir)
            print(f"📋 Available files in models directory: {available_files}")
            
            # Load Chili model
            if os.path.exists(CHILI_MODEL_PATH):
                print(f"🔧 DEBUG: Chili model found at: {CHILI_MODEL_PATH}")
                try:
                    # FIX: Add weights_only=False for PyTorch 2.6 compatibility
                    chili_model = torch.load(CHILI_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
                    chili_model.eval()
                    loaded_models['chili'] = True
                    print("✅ Chili model loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load chili model: {e}")
            else:
                print(f"❌ Chili model not found at: {CHILI_MODEL_PATH}")

            # Load Eggplant model  
            if os.path.exists(EGGPLANT_MODEL_PATH):
                print(f"🔧 DEBUG: Eggplant model found at: {EGGPLANT_MODEL_PATH}")
                try:
                    # FIX: Add weights_only=False for PyTorch 2.6 compatibility
                    eggplant_model = torch.load(EGGPLANT_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
                    eggplant_model.eval()
                    loaded_models['eggplant'] = True
                    print("✅ Eggplant model loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load eggplant model: {e}")
            else:
                print(f"❌ Eggplant model not found at: {EGGPLANT_MODEL_PATH}")

            # Load Tomato Leaf model
            if os.path.exists(TOMATO_LEAF_MODEL_PATH):
                print(f"🔧 DEBUG: Tomato leaf model found at: {TOMATO_LEAF_MODEL_PATH}")
                try:
                    # FIX: Add weights_only=False for PyTorch 2.6 compatibility
                    tomato_leaf_model = torch.load(TOMATO_LEAF_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
                    tomato_leaf_model.eval()
                    loaded_models['tomato_leaf'] = True
                    print("✅ Tomato leaf model loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load tomato leaf model: {e}")
            else:
                print(f"❌ Tomato leaf model not found at: {TOMATO_LEAF_MODEL_PATH}")

            # Load Tomato Vegetation model
            if os.path.exists(TOMATO_VEGE_MODEL_PATH):
                print(f"🔧 DEBUG: Tomato vegetation model found at: {TOMATO_VEGE_MODEL_PATH}")
                try:
                    tomato_vege_model = keras.models.load_model(TOMATO_VEGE_MODEL_PATH)
                    loaded_models['tomato_vege'] = True
                    print("✅ Tomato vegetation model loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load tomato vegetation model: {e}")
            else:
                print(f"❌ Tomato vegetation model not found at: {TOMATO_VEGE_MODEL_PATH}")

    except Exception as e:
        print(f"💥 CRITICAL ERROR during model loading: {e}")
else:
    # TEST MODE - Simulate successful model loading
    print("🧪 ============================================")
    print("🧪 RUNNING IN TEST MODE - Using dummy models")
    print("🧪 Set TEST_MODE = False to use real models")
    print("🧪 ============================================")
    loaded_models = {
        'chili': True,
        'eggplant': True, 
        'tomato_leaf': True,
        'tomato_vege': True
    }

# Update the models_loaded_successfully flag
models_loaded_successfully = any(loaded_models.values())
print(f"📊 Models loaded successfully: {models_loaded_successfully}")
print(f"🔍 Loaded models status: {loaded_models}")

# Define preprocessing steps for different models
def preprocess_image_torch(image_path):
    print(f"Preprocessing image for PyTorch: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error during PyTorch preprocessing: {e}")
        return None

def preprocess_image_keras(image_path, target_size=(224, 224)):
    print(f"Preprocessing image for Keras: {image_path}")
    try:
        img = keras.utils.load_img(image_path, target_size=target_size)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0  # Simple normalization for generic models
    except Exception as e:
        print(f"Error during Keras preprocessing: {e}")
        return None

# Function to make predictions with PyTorch models
def predict_pytorch(model, image_tensor, class_names):
    print("Making prediction with PyTorch model...")
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(probabilities, 0)
            predicted_label = class_names[predicted_idx.item()]
            confidence = probabilities[predicted_idx.item()].item()
            print(f"PyTorch Prediction: {predicted_label}, Confidence: {confidence}")
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during PyTorch prediction: {e}")
        return None, None

# Function to make predictions with Keras models
def predict_keras(model, processed_image, class_names):
    print("Making prediction with Keras model...")
    try:
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = class_names[predicted_idx]
        confidence = predictions[0][predicted_idx]
        print(f"Keras Prediction: {predicted_label}, Confidence: {confidence}")
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during Keras prediction: {e}")
        return None, None

@login_required
def scan_view(request):
    """
    Renders the scan page. This view is for GET requests.
    """
    return render(request, 'detection/Scanpage.html')

def debug_database(request):
    """Temporary view to debug database content"""
    plants = Plant.objects.all()
    result = {}
    
    for plant in plants:
        diseases = Disease.objects.filter(plants=plant)
        result[plant.name] = [{
            'name': disease.name,
            'name_fil': disease.get_translated_name('tl'),
            'description': disease.description[:50] + '...' if disease.description else 'None',
            'description_fil': disease.get_translated_description('tl')[:50] + '...' if disease.get_translated_description('tl') else 'None',
        } for disease in diseases]
    
    return JsonResponse(result)

def disease_list_view(request):
    """
    View for the disease list page with translation support
    """
    plants_with_diseases = []
    current_language = get_language()
    
    print(f"🌐 Loading disease list in language: {current_language}")
    
    for plant in Plant.objects.all().prefetch_related('diseases'):
        diseases = plant.diseases.all()
        
        # Get translated plant name
        if current_language == 'tl':
            plant_name = plant.get_translated_name('tl')
        else:
            plant_name = plant.name
            
        plant_data = {
            'name': plant_name,
            'original_name': plant.name,  # Keep original for reference
            'diseases': []
        }
        
        for disease in diseases:
            # Get translated disease information
            if current_language == 'tl':
                disease_data = {
                    'id': disease.id,
                    'name': disease.get_translated_name('tl'),
                    'description': disease.get_translated_description('tl'),
                    'symptoms': disease.get_translated_symptoms('tl'),
                    'prevention': disease.get_translated_prevention('tl'),
                    'original_name': disease.name,  # Keep original for reference
                    'translation_status': disease.translation_status(),
                }
            else:
                disease_data = {
                    'id': disease.id,
                    'name': disease.name,
                    'description': disease.description,
                    'symptoms': disease.symptoms,
                    'prevention': disease.prevention,
                    'original_name': disease.name,
                    'translation_status': disease.translation_status(),
                }
            
            plant_data['diseases'].append(disease_data)
        
        plants_with_diseases.append(plant_data)
    
    context = {
        'plants_with_diseases': plants_with_diseases,
        'current_language': current_language,
    }
    
    print(f"📋 Loaded {len(plants_with_diseases)} plants with diseases")
    return render(request, 'detection/list_page.html', context)

@login_required
@csrf_exempt
def detect_view(request):
    """
    Handles image detection and saves the scan result to the database.
    """
    if request.method == 'POST':
        try:
            print("📨 Received POST request to detect_view")
            
            # =============================================================================
            # FIXED LANGUAGE DETECTION - Get language from multiple sources
            # =============================================================================
            current_language = get_language()
            
            # Check if language is explicitly passed in POST data (frontend should send this)
            post_language = request.POST.get('language', '')
            if post_language:
                current_language = post_language
                print(f"🌐 Using language from POST data: {current_language}")
            else:
                print(f"🌐 Using language from Django context: {current_language}")
            
            # Also check Accept-Language header as fallback
            accept_language = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
            if not current_language and accept_language:
                # Extract first language from header (e.g., "tl,en;q=0.9" -> "tl")
                primary_lang = accept_language.split(',')[0].split('-')[0].lower()
                if primary_lang in ['en', 'tl']:
                    current_language = primary_lang
                    print(f"🌐 Using language from header: {current_language}")
            
            # Final fallback
            if not current_language:
                current_language = 'en'
                print(f"🌐 Using default language: {current_language}")
            
            # IMPORTANT: Check if we're in test mode
            if TEST_MODE:
                print("🧪 Processing request in TEST MODE")
                print(f"👤 Authenticated user: {request.user.username}")

                # Get request data
                plant_type_str = request.POST.get('plant_type', '').lower()
                uploaded_image = request.FILES.get('image')

                print(f"🌱 Plant type: {plant_type_str}")
                print(f"🖼️ Uploaded image: {uploaded_image}")

                if not uploaded_image:
                    return JsonResponse({'error': 'No image file provided'}, status=400)
                if not plant_type_str:
                    return JsonResponse({'error': 'No plant type provided'}, status=400)

                # TEST MODE PREDICTIONS
                import random
                available_predictions = TEST_PREDICTIONS.get(plant_type_str, [])
                if available_predictions:
                    prediction_label, confidence, description, symptoms, prevention = random.choice(available_predictions)
                    print(f"🧪 Test prediction: {prediction_label} (confidence: {confidence})")
                else:
                    prediction_label = "No test data available"
                    confidence = 0.0
                    description = "Test mode is active but no test predictions configured for this plant type."
                    symptoms = "N/A"
                    prevention = "N/A"

                # =============================================================================
                # TEST MODE TRANSLATION HANDLING
                # =============================================================================
                
                # Map frontend plant type strings to database Plant names
                plant_mapping = {
                    'chili': 'Chili Pepper',
                    'eggplant': 'Eggplant', 
                    'tomato': 'Tomato'
                }

                db_plant_name = plant_mapping.get(plant_type_str)
                plant_obj = None
                disease_obj = None
                
                if db_plant_name:
                    try:
                        plant_obj = Plant.objects.get(name=db_plant_name)
                        print(f"✅ Found plant object: {plant_obj.name} (ID: {plant_obj.id})")
                        
                        # Try to find matching disease in database for translation
                        try:
                            disease_obj = Disease.objects.get(
                                name__iexact=prediction_label,
                                plants=plant_obj
                            )
                            print(f"✅ Found disease object for translation: {disease_obj.name}")
                        except Disease.DoesNotExist:
                            print(f"⚠️ No disease object found for '{prediction_label}' - using default text")
                        except Exception as e:
                            print(f"⚠️ Error looking up disease: {e}")
                            
                    except Plant.DoesNotExist:
                        print(f"❌ Plant '{db_plant_name}' not found in database")
                else:
                    print(f"❌ Unknown plant type from mapping: {plant_type_str}")

                # Apply translations if disease object found and language is Filipino
                final_disease_name = prediction_label
                final_description = description
                final_symptoms = symptoms
                final_prevention = prevention

                if current_language == 'tl' and disease_obj:
                    print("🌐 Applying Filipino translations from database")
                    final_disease_name = disease_obj.get_translated_name('tl') or prediction_label
                    final_description = disease_obj.get_translated_description('tl') or description
                    final_symptoms = disease_obj.get_translated_symptoms('tl') or symptoms
                    final_prevention = disease_obj.get_translated_prevention('tl') or prevention
                    
                    # Debug translation results
                    print(f"🔍 TRANSLATION RESULTS:")
                    print(f"   Disease: {final_disease_name}")
                    print(f"   Description length: {len(final_description)}")
                    print(f"   Symptoms length: {len(final_symptoms)}")
                    print(f"   Prevention length: {len(final_prevention)}")
                else:
                    print(f"🌐 Using default English text (Language: {current_language}, Disease Obj: {disease_obj is not None})")

                # Create Scan record in test mode too
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
                        print(f"✅ Scan record saved for user {request.user.username}")
                    except Exception as e:
                        print(f"❌ Error saving Scan record: {e}")

                # Return test mode response with proper translations
                final_response_data = {
                    'disease': final_disease_name,
                    'confidence': f"{confidence*100:.2f}%" if confidence else "N/A",
                    'description': final_description,
                    'symptoms': final_symptoms,
                    'prevention': final_prevention,
                    'test_mode': True,
                    'language': current_language,
                    'note': "This result is from test mode - using simulated predictions"
                }

                print(f"📤 Returning TEST response in {current_language}: {final_response_data}")
                return JsonResponse(final_response_data)
            else:
                # REAL MODE PROCESSING
                print("🔍 Processing request with REAL MODELS")
                print(f"👤 Authenticated user: {request.user.username}")

                # Check if any models were loaded successfully
                if not models_loaded_successfully:
                    print("❌ No models loaded - returning error")
                    return JsonResponse({
                        'error': 'No machine learning models are available on the server. Please contact administrator.',
                        'details': 'Server cannot process scans without model files.'
                    }, status=503)

                # Get request data
                plant_type_str = request.POST.get('plant_type', '').lower()
                uploaded_image = request.FILES.get('image')

                print(f"🌱 Plant type: {plant_type_str}")
                print(f"🖼️ Uploaded image: {uploaded_image}")

                if not uploaded_image:
                    return JsonResponse({'error': 'No image file provided'}, status=400)
                if not plant_type_str:
                    return JsonResponse({'error': 'No plant type provided'}, status=400)

                # Check if specific model for this plant type is available
                plant_model_available = False
                if plant_type_str == 'eggplant' and loaded_models['eggplant']:
                    plant_model_available = True
                elif plant_type_str == 'chili' and loaded_models['chili']:
                    plant_model_available = True
                elif plant_type_str == 'tomato' and (loaded_models['tomato_leaf'] or loaded_models['tomato_vege']):
                    plant_model_available = True

                if not plant_model_available:
                    return JsonResponse({
                        'error': f'Model for {plant_type_str} is not available',
                        'details': 'The specific model required for this plant type could not be loaded.'
                    }, status=503)

                # Save the uploaded file temporarily
                temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_image.name)
                saved_file_name = default_storage.save(temp_file_path, uploaded_image)
                full_file_path = default_storage.path(saved_file_name)
                print(f"💾 Image temporarily saved to: {full_file_path}")

                # REAL MODEL PREDICTIONS
                print("🔍 Using REAL model predictions")
                prediction_label = "No disease detected"
                confidence = 0.0
                
                if plant_type_str == 'eggplant' and eggplant_model:
                    eggplant_class_names = ["Aphids", "Cercospora Leaf Spot", "Defect Eggplant", "Flea Beetles",
                                          "Fresh Eggplant", "Fresh Eggplant Leaf", "Leaf Wilt", "Phytophthora Blight",
                                          "Powdery Mildew", "Tobacco Mosaic Virus"]
                    processed_image = preprocess_image_torch(full_file_path)
                    if processed_image is not None:
                        prediction, conf = predict_pytorch(eggplant_model, processed_image, eggplant_class_names)
                        if prediction:
                            prediction_label = prediction
                            confidence = conf
                            print(f"Eggplant model raw prediction: {prediction_label}, confidence: {confidence}")

                elif plant_type_str == 'chili' and chili_model:
                    chili_class_names = ["Damaged", "Dried", "Old", "Ripe", "Unripe"]
                    processed_image = preprocess_image_torch(full_file_path)
                    if processed_image is not None:
                        prediction, conf = predict_pytorch(chili_model, processed_image, chili_class_names)
                        if prediction:
                            prediction_label = prediction
                            confidence = conf
                            print(f"Chili model raw prediction: {prediction_label}, confidence: {confidence}")

                elif plant_type_str == 'tomato':
                    # Prioritize disease detection from the leaf model first
                    if tomato_leaf_model:
                        tomato_leaf_class_names = ["Bacterial Spot", "Early Blight", "Healthy Tomato Leaves", "Late Blight",
                                                 "Leaf Mold", "Powdery Mildew", "Septoria Leaf Spot", "Spider Mites",
                                                 "Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"]
                        processed_image = preprocess_image_torch(full_file_path)
                        if processed_image is not None:
                            leaf_prediction, leaf_conf = predict_pytorch(tomato_leaf_model, processed_image, tomato_leaf_class_names)
                            print(f"Tomato leaf model raw prediction: {leaf_prediction}, confidence: {leaf_conf}")

                            # If the prediction is NOT "Healthy Tomato Leaves", it's a disease.
                            if leaf_prediction and leaf_prediction != "Healthy Tomato Leaves":
                                prediction_label = leaf_prediction
                                confidence = leaf_conf
                            else:
                                # If it's healthy, we can check the vegetation model for other labels
                                if tomato_vege_model:
                                    tomato_vege_class_names = ["Damaged", "Old", "Ripe", "Unripe", "Healthy"]
                                    vege_processed_image = preprocess_image_keras(full_file_path)
                                    if vege_processed_image is not None:
                                        vege_prediction, vege_conf = predict_keras(tomato_vege_model, vege_processed_image, tomato_vege_class_names)
                                        prediction_label = vege_prediction
                                        confidence = vege_conf
                                        print(f"Tomato vegetation model raw prediction: {vege_prediction}, confidence: {vege_conf}")
                                else:
                                    prediction_label = "Tomato leaf model detected healthy, but vegetation model not loaded."
                                    confidence = leaf_conf  # Use leaf model confidence if no other prediction
                        else:
                            prediction_label = "Tomato leaf model preprocessing failed."
                    else:
                        prediction_label = "Tomato leaf model not loaded."
                        # Fallback to vegetation model if leaf model is not available
                        if tomato_vege_model:
                            tomato_vege_class_names = ["Damaged", "Old", "Ripe", "Unripe", "Healthy"]
                            vege_processed_image = preprocess_image_keras(full_file_path)
                            if vege_processed_image is not None:
                                vege_prediction, vege_conf = predict_keras(tomato_vege_model, vege_processed_image, tomato_vege_class_names)
                                prediction_label = vege_prediction
                                confidence = vege_conf
                                print(f"Tomato vegetation model raw prediction (fallback): {vege_prediction}, confidence: {vege_conf}")
                            else:
                                prediction_label = "Tomato vegetation model preprocessing failed (fallback)."
                        else:
                            prediction_label = "No tomato models available."
                else:
                    prediction_label = "No model available for selected plant type."
                    print(f"No model available for plant type: {plant_type_str}")

                # Clean up temporary file
                if 'saved_file_name' in locals() and default_storage.exists(saved_file_name):
                    default_storage.delete(saved_file_name)
                    print("✅ Temporary file cleaned up")

                # =============================================================================
                # ENHANCED DISEASE LOOKUP WITH TRANSLATION SUPPORT
                # =============================================================================

                # Map frontend plant type strings to database Plant names
                plant_mapping = {
                    'chili': 'Chili Pepper',
                    'eggplant': 'Eggplant', 
                    'tomato': 'Tomato'
                }

                db_plant_name = plant_mapping.get(plant_type_str)
                plant_obj = None
                if db_plant_name:
                    try:
                        plant_obj = Plant.objects.get(name=db_plant_name)
                        print(f"✅ Found plant object: {plant_obj.name} (ID: {plant_obj.id})")
                        
                        # DEBUG: Show all diseases for this plant
                        all_diseases_for_plant = Disease.objects.filter(plants=plant_obj)
                        print(f"🔍 ALL DISEASES FOR {plant_obj.name}:")
                        for disease in all_diseases_for_plant:
                            print(f"   - {disease.name} (ID: {disease.id})")
                            
                    except Plant.DoesNotExist:
                        print(f"❌ Plant '{db_plant_name}' not found in database")
                else:
                    print(f"❌ Unknown plant type from mapping: {plant_type_str}")

                # Determine if it's a disease and get disease info from database
                disease_obj = None
                disease_description = "No description available."
                disease_symptoms = "No symptoms information available." 
                disease_prevention = "No prevention strategies available."

                print(f"🔍 LOOKING FOR DISEASE: '{prediction_label}' for plant: '{db_plant_name}'")

                # First, try to find the exact disease name in the database for this specific plant
                if plant_obj:
                    try:
                        # DEBUG: Try multiple search strategies
                        print(f"🔍 SEARCH STRATEGIES:")
                        
                        # Strategy 1: Exact case-insensitive match
                        exact_match = Disease.objects.filter(
                            name__iexact=prediction_label,
                            plants=plant_obj
                        ).first()
                        print(f"  1. Exact match: {exact_match.name if exact_match else 'NOT FOUND'}")
                        
                        if exact_match:
                            disease_obj = exact_match
                            print(f"✅ FOUND EXACT MATCH: {disease_obj.name}")
                        
                        # Strategy 2: Remove spaces and compare (for "LeafWilt" vs "Leaf Wilt")
                        if not disease_obj:
                            prediction_no_spaces = prediction_label.replace(' ', '')
                            no_spaces_match = Disease.objects.filter(
                                plants=plant_obj
                            ).extra(
                                where=["REPLACE(name, ' ', '') ILIKE %s"],
                                params=[f"%{prediction_no_spaces}%"]
                            ).first()
                            print(f"  2. No spaces match: {no_spaces_match.name if no_spaces_match else 'NOT FOUND'}")
                            
                            if no_spaces_match:
                                disease_obj = no_spaces_match
                                print(f"✅ FOUND NO-SPACES MATCH: {disease_obj.name}")
                        
                        # Strategy 3: Partial match
                        if not disease_obj:
                            partial_match = Disease.objects.filter(
                                name__icontains=prediction_label,
                                plants=plant_obj
                            ).first()
                            print(f"  3. Partial match: {partial_match.name if partial_match else 'NOT FOUND'}")
                            
                            if partial_match:
                                disease_obj = partial_match
                                print(f"✅ FOUND PARTIAL MATCH: {disease_obj.name}")
                        
                        # Strategy 4: First word match
                        if not disease_obj and ' ' in prediction_label:
                            first_word = prediction_label.split()[0]
                            first_word_match = Disease.objects.filter(
                                name__icontains=first_word,
                                plants=plant_obj
                            ).first()
                            print(f"  4. First word match: {first_word_match.name if first_word_match else 'NOT FOUND'}")
                            
                            if first_word_match:
                                disease_obj = first_word_match
                                print(f"✅ FOUND FIRST WORD MATCH: {disease_obj.name}")
                        
                        # Strategy 5: Common name variations
                        if not disease_obj:
                            # Handle common naming variations
                            common_variations = {
                                'leafwilt': 'Leaf Wilt',
                                'powderymildew': 'Powdery Mildew',
                                'earlyblight': 'Early Blight', 
                                'lateblight': 'Late Blight',
                                'bacterialspot': 'Bacterial Spot',
                                'bacterialwilt': 'Bacterial Wilt',
                                'leafspot': 'Cercospora Leaf Spot',
                                'mosaicvirus': 'Tomato Mosaic Virus',
                                'yellowleafcurl': 'Tomato Yellow Leaf Curl Virus',
                                'septorialeafspot': 'Septoria Leaf Spot',
                                'tomatomosaicvirus': 'Tomato Mosaic Virus',
                                'tomatoyellowleafcurlvirus': 'Tomato Yellow Leaf Curl Virus',
                                'phytophthora': 'Phytophthora Blight',
                                'cercospora': 'Cercospora Leaf Spot',
                                'septoria': 'Septoria Leaf Spot',
                                'targetspot': 'Target Spot',
                                'leafmold': 'Leaf Mold',
                                'spidermites': 'Spider Mites',
                            }
                            
                            prediction_lower_no_spaces = prediction_label.lower().replace(' ', '')
                            if prediction_lower_no_spaces in common_variations:
                                variation_match = Disease.objects.filter(
                                    name__iexact=common_variations[prediction_lower_no_spaces],
                                    plants=plant_obj
                                ).first()
                                print(f"  5. Common variation match: {variation_match.name if variation_match else 'NOT FOUND'}")
                                
                                if variation_match:
                                    disease_obj = variation_match
                                    print(f"✅ FOUND COMMON VARIATION MATCH: {disease_obj.name}")
                        
                        # If we found a disease object, use its data
                        if disease_obj:
                            print(f"✅ USING DATABASE DISEASE: {disease_obj.name}")
                            
                            # Get translations based on current language
                            if current_language == 'tl':
                                disease_name = disease_obj.get_translated_name('tl') or disease_obj.name
                                disease_description = disease_obj.get_translated_description('tl') or disease_obj.description or disease_description
                                disease_symptoms = disease_obj.get_translated_symptoms('tl') or disease_obj.symptoms or disease_symptoms
                                disease_prevention = disease_obj.get_translated_prevention('tl') or disease_obj.prevention or disease_prevention
                                print(f"🌐 Using Filipino translations for disease: {disease_name}")
                            else:
                                disease_name = disease_obj.name
                                disease_description = disease_obj.description or disease_description
                                disease_symptoms = disease_obj.symptoms or disease_symptoms
                                disease_prevention = disease_obj.prevention or disease_prevention
                                print(f"🌐 Using English content for disease: {disease_name}")
                            
                        else:
                            print(f"❌ NO DATABASE MATCH FOUND for '{prediction_label}'")
                            # Use original prediction label
                            disease_name = prediction_label
                            
                    except Exception as e:
                        print(f"❌ Error during disease lookup: {e}")
                        disease_name = prediction_label

                # If no disease object found, check if it's a non-disease condition
                non_disease_labels = {
                    "Damaged", "Dried", "Old", "Ripe", "Unripe",
                    "Defect Eggplant", "Fresh Eggplant", "Fresh Eggplant Leaf", 
                    "Healthy Tomato Leaves", "Healthy", "No disease detected",
                    "Chili Pepper", "Eggplant", "Tomato",
                }

                is_disease = prediction_label not in non_disease_labels
                print(f"🔍 IS DISEASE CHECK: '{prediction_label}' -> {is_disease}")

                if not disease_obj and is_disease:
                    # Create a new disease entry if it's a real disease but not in database
                    try:
                        disease_obj, created = Disease.objects.get_or_create(
                            name=prediction_label,
                            defaults={
                                'description': f"Description for {prediction_label} will be added soon.",
                                'symptoms': f"Symptoms information for {prediction_label} is being updated.",
                                'prevention': f"Prevention methods for {prediction_label} will be provided soon."
                            }
                        )
                        if created and plant_obj:
                            disease_obj.plants.add(plant_obj)
                            # Auto-translate the new disease
                            disease_obj.auto_translate_all()
                            print(f"🆕 CREATED NEW DISEASE ENTRY: {prediction_label}")
                    except Exception as e:
                        print(f"❌ Error creating new disease entry: {e}")
                elif not is_disease:
                    # Handle non-disease labels with appropriate messages
                    print(f"🔍 HANDLING NON-DISEASE: {prediction_label}")
                    if prediction_label == "Healthy Tomato Leaves" or prediction_label == "Healthy":
                        disease_description = "The plant appears healthy based on the scan."
                        disease_symptoms = "No symptoms of disease detected."
                        disease_prevention = "Continue good agricultural practices."
                    elif prediction_label == "No disease detected":
                        disease_description = "The scan did not detect any specific disease."
                        disease_symptoms = "No specific symptoms identified." 
                        disease_prevention = "Monitor the plant for any changes."
                    else:
                        disease_description = f"The scan identified the plant as: {prediction_label}. This is a general condition, not a disease."
                        disease_symptoms = "This condition indicates the plant's growth stage or quality, not disease symptoms."
                        disease_prevention = "No specific disease prevention needed. Focus on general plant care practices."

                # Create Scan record
                if plant_obj:
                    try:
                        Scan.objects.create(
                            user=request.user,
                            plant=plant_obj,
                            disease=disease_obj if is_disease else None,
                            scanned_image=uploaded_image,
                            detected_result=prediction_label,
                            confidence=confidence if confidence else 0.0,
                        )
                        print(f"✅ Scan record saved for user {request.user.username}")
                    except Exception as e:
                        print(f"❌ Error saving Scan record: {e}")

                # Use the translated disease name if available
                final_disease_name = disease_obj.get_translated_name(current_language) if disease_obj else prediction_label

                # Debug: Show what data is being returned
                print(f"🔍 FINAL DATA BEING RETURNED:")
                print(f"   Language: {current_language}")
                print(f"   Disease: {final_disease_name}")
                print(f"   Confidence: {confidence}")
                print(f"   Description: {disease_description[:100]}...")
                print(f"   Symptoms: {disease_symptoms[:100]}...") 
                print(f"   Prevention: {disease_prevention[:100]}...")
                print(f"   Disease Object Found: {disease_obj is not None}")
                if disease_obj:
                    print(f"   Disease DB ID: {disease_obj.id}")

                # Return response with appropriate language content
                final_response_data = {
                    'disease': final_disease_name,
                    'confidence': f"{confidence*100:.2f}%" if confidence else "N/A",
                    'description': disease_description,
                    'symptoms': disease_symptoms,
                    'prevention': disease_prevention,
                    'language': current_language,  # Include language info for frontend
                }

                print(f"📤 Returning REAL response in {current_language}: {final_response_data}")
                return JsonResponse(final_response_data)

        except Exception as e:
            print(f"💥 Error in detect_view: {str(e)}")
            # Clean up if file was created
            if 'saved_file_name' in locals() and default_storage.exists(saved_file_name):
                default_storage.delete(saved_file_name)
            return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)