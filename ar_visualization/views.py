from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os
import json
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

@login_required(login_url="login_view")
def ar_page(request):
    return render(request, 'ar_visualization/ARpage.html')

@login_required(login_url="login_view")
def get_crop_model(request, crop_type):
    crop_models = {
        'eggplant': {
            'model_url': os.path.join(settings.STATIC_URL, 'ar_models/eggplant.glb'),
            'scale': 0.5,
            'info': 'Eggplant 3D Model'
        },
        'tomato': {
            'model_url': os.path.join(settings.STATIC_URL, 'ar_models/crop_tomatoes.glb'),
            'scale': 3.5,
            'info': 'Tomato Plant 3D Model'
        },
        'chili': {
            'model_url': os.path.join(settings.STATIC_URL, 'ar_models/chili_pepper.glb'),
            'scale': 0.8,
            'info': 'Chili Pepper Plant 3D Model'
        },
    }
    return JsonResponse(crop_models.get(crop_type.lower(), {
        'model_url': os.path.join(settings.STATIC_URL, 'ar_models/default.glb'),
        'scale': 0.5,
        'info': 'Default Plant Model'
    }))

@login_required(login_url="login_view")
def get_crop_labels(request, crop_type):
    """
    Returns label data for a specific crop type.
    EDIT THIS PART to customize your labels and arrow positions
    """
    crop_labels = {
        'eggplant': {
            'parts': [
                {
                    # LABEL TEXT - Change these:
                    'name': 'Leaves',                    # Label title
                    'diseases': ['Leaf Spot', 'Powdery Mildew', 'Blight'],  # Diseases list
                    
                    # LABEL POSITION - Change these (pixels from top-left):
                    'labelPosition': {'x': 50, 'y': 100},
                    
                    # ARROW TARGET - Change these to point arrows (3D coordinates):
                    'targetPosition': {'x': 1, 'y': 0.5, 'z': 0.5}
                },
                {
                    'name': 'Stem',                      # Change this text
                    'diseases': ['Stem Rot', 'Wilt Disease'],    # Change these diseases
                    'labelPosition': {'x': 50, 'y': 300}, # Change position
                    'targetPosition': {'x': 1, 'y': 0, 'z': -1} # Change arrow target
                },
                {
                    'name': 'Fruit',                     # Your text here
                    'diseases': ['Fruit Rot', 'Anthracnose'],           # Your diseases here
                    'labelPosition': {'x': 50, 'y': 500}, # Your position
                    'targetPosition': {'x': 0, 'y': -0.5, 'z': 0.5} # Your target
                }
            ]
        },
        'tomato': {
            'parts': [
                {
                    'name': 'Leaves',
                    'diseases': ['Early Blight', 'Late Blight', 'Leaf Mold'],
                    'labelPosition': {'x': 50, 'y': 100},
                    'targetPosition': {'x': 0, 'y': 2, 'z': 0}
                },
                {
                    'name': 'Stem',
                    'diseases': ['Stem Canker', 'Bacterial Wilt'],
                    'labelPosition': {'x': 50, 'y': 300},
                    'targetPosition': {'x': 0, 'y': 1, 'z': 0}
                },
                {
                    'name': 'Fruit',
                    'diseases': ['Blossom End Rot', 'Tomato Spotted Wilt'],
                    'labelPosition': {'x': 50, 'y': 500},
                    'targetPosition': {'x': 0, 'y': 0, 'z': 0.5}
                }
            ]
        },
        'chili': {
            'parts': [
                {
                    'name': 'Leaves',
                    'diseases': ['Bacterial Leaf Spot', 'Powdery Mildew'],
                    'labelPosition': {'x': 50, 'y': 100},
                    'targetPosition': {'x': 0, 'y': 1.2, 'z': 0}
                },
                {
                    'name': 'Stem',
                    'diseases': ['Stem Rot', 'Phytophthora Blight'],
                    'labelPosition': {'x': 50, 'y': 300},
                    'targetPosition': {'x': 0, 'y': 0.6, 'z': 0}
                },
                {
                    'name': 'Fruit',
                    'diseases': ['Anthracnose', 'Sunscald'],
                    'labelPosition': {'x': 50, 'y': 500},
                    'targetPosition': {'x': 0, 'y': 0, 'z': 0.3}
                }
            ]
        }
    }
    
    return JsonResponse(crop_labels.get(crop_type.lower(), {'parts': []}))

@login_required(login_url="login_view")
@csrf_exempt
def save_crop_labels(request, crop_type):
    """
    API endpoint to save custom labels (if you want admin-editable labels)
    This requires authentication and should be protected.
    """
    if request.method == 'POST':
        try:
            labels_data = json.loads(request.body)
            # Here you would typically save to database
            # For now, we'll just return success
            return JsonResponse({'status': 'success', 'message': 'Labels saved successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@login_required(login_url="login_view")
def ar_camera_view(request):
    crop_type = request.GET.get('crop', '')
    return render(request, 'ar_visualization/ar_camera.html', {
        'selected_crop': crop_type,
    })

@login_required(login_url="login_view")
def scan_page(request):
    return render(request, 'detection/Scanpage.html')