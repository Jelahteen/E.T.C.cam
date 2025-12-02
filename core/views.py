from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.translation import get_language
import os
from PIL import Image
import torch
from torchvision import transforms
from django.core.files.storage import default_storage
from django.conf import settings
import random

from .models import Plant, Disease, Scan
from django.contrib.auth import get_user_model

User = get_user_model()

# TEST MODE
TEST_MODE = True  # Change to False for real models

# TEMP DIR
TEMP_UPLOAD_DIR = 'temp_uploads'
os.makedirs(os.path.join(settings.BASE_DIR, TEMP_UPLOAD_DIR), exist_ok=True)

# MODEL PATHS
CHILI_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'chili_resnet.pth')
EGGPLANT_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'eggplant_resnet.pth')
TOMATO_LEAF_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'models', 'tomato_leaf_resnet.pth')

# INITIALIZE MODELS
chili_model = None
eggplant_model = None
tomato_leaf_model = None

loaded_models = {'chili': False, 'eggplant': False, 'tomato_leaf': False}

if not TEST_MODE:
    if os.path.exists(CHILI_MODEL_PATH):
        chili_model = torch.load(CHILI_MODEL_PATH, map_location='cpu', weights_only=False)
        chili_model.eval()
        loaded_models['chili'] = True
    if os.path.exists(EGGPLANT_MODEL_PATH):
        eggplant_model = torch.load(EGGPLANT_MODEL_PATH, map_location='cpu', weights_only=False)
        eggplant_model.eval()
        loaded_models['eggplant'] = True
    if os.path.exists(TOMATO_LEAF_MODEL_PATH):
        tomato_leaf_model = torch.load(TOMATO_LEAF_MODEL_PATH, map_location='cpu', weights_only=False)
        tomato_leaf_model.eval()
        loaded_models['tomato_leaf'] = True
else:
    loaded_models = {'chili': True, 'eggplant': True, 'tomato_leaf': True}

# PREPROCESSING
def preprocess_image_torch(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def predict_pytorch(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        _, idx = torch.max(probs, 0)
        return class_names[idx.item()], probs[idx.item()]

# VIEWS
@login_required
def scan_view(request):
    return render(request, 'detection/Scanpage.html')

@login_required
@csrf_exempt
def detect_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

    try:
        current_language = request.POST.get('language', get_language()) or 'en'
        plant_type = request.POST.get('plant_type', '').lower()
        uploaded_image = request.FILES.get('image')

        if not plant_type or not uploaded_image:
            return JsonResponse({'error': 'Missing plant type or image.'}, status=400)

        # SAVE TEMP FILE
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_image.name)
        saved_file_name = default_storage.save(temp_file_path, uploaded_image)
        full_file_path = default_storage.path(saved_file_name)

        # TEST MODE PREDICTION
        if TEST_MODE:
            TEST_PREDICTIONS = {
                'eggplant': [('Fresh Eggplant', 0.95)],
                'chili': [('Ripe', 0.92)],
                'tomato': [('Healthy Tomato Leaves', 0.94)]
            }
            prediction, confidence = random.choice(TEST_PREDICTIONS.get(plant_type, [('Unknown', 0.0)]))

        # REAL MODE PREDICTION
        else:
            prediction = "No model available"
            confidence = 0.0

            if plant_type == 'eggplant' and eggplant_model:
                class_names = ["Aphids","Cercospora Leaf Spot","Defect Eggplant","Flea Beetles",
                               "Fresh Eggplant","Leaf Wilt","Powdery Mildew","Phytophthora Blight",
                               "Tobacco Mosaic Virus"]
                img_tensor = preprocess_image_torch(full_file_path)
                prediction, confidence = predict_pytorch(eggplant_model, img_tensor, class_names)

            elif plant_type == 'chili' and chili_model:
                class_names = ["Damaged","Dried","Old","Ripe","Unripe"]
                img_tensor = preprocess_image_torch(full_file_path)
                prediction, confidence = predict_pytorch(chili_model, img_tensor, class_names)

            elif plant_type == 'tomato' and tomato_leaf_model:
                class_names = ["Bacterial Spot","Early Blight","Healthy Tomato Leaves","Late Blight",
                               "Leaf Mold","Powdery Mildew","Septoria Leaf Spot","Spider Mites",
                               "Target Spot","Tomato Mosaic Virus","Tomato Yellow Leaf Curl Virus"]
                img_tensor = preprocess_image_torch(full_file_path)
                leaf_pred, leaf_conf = predict_pytorch(tomato_leaf_model, img_tensor, class_names)
                if leaf_pred != "Healthy Tomato Leaves":
                    prediction = leaf_pred
                    confidence = leaf_conf
                else:
                    prediction = "Healthy"
                    confidence = 1.0

        # CLEANUP
        if default_storage.exists(saved_file_name):
            default_storage.delete(saved_file_name)

        # GET PLANT AND DISEASE OBJECTS
        plant_mapping = {'chili':'Chili Pepper','eggplant':'Eggplant','tomato':'Tomato'}
        db_plant_name = plant_mapping.get(plant_type)
        plant_obj = Plant.objects.filter(name=db_plant_name).first()
        disease_obj = Disease.objects.filter(name__iexact=prediction, plants=plant_obj).first() if plant_obj else None

        # SAVE SCAN
        if plant_obj:
            Scan.objects.create(
                user=request.user,
                plant=plant_obj,
                disease=disease_obj if disease_obj else None,
                scanned_image=uploaded_image,
                detected_result=prediction,
                confidence=confidence
            )

        response_data = {
            'disease': prediction,
            'confidence': f"{confidence*100:.2f}%" if confidence else "N/A",
            'description': disease_obj.description if disease_obj else "No description available.",
            'symptoms': disease_obj.symptoms if disease_obj else "No symptoms available.",
            'prevention': disease_obj.prevention if disease_obj else "No prevention available.",
            'language': current_language
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
