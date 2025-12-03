from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.utils.translation import gettext_lazy as _
from django.utils.translation import get_language
from django.http import JsonResponse
import json
from datetime import datetime, timedelta
from django.utils import timezone

from .forms import SimpleUserCreationForm, LoginForm
from detection.models import Scan, Plant, Disease 
from django.contrib.auth import get_user_model

User = get_user_model()

def homepage(request):
    """
    Renders the homepage.
    """
    return render(request, 'core/homepage.html')

def login_view(request):
    """
    Handles user login using the custom LoginForm.
    """
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard_page')
            else:
                form.add_error(None, _('Invalid username or password.'))
    else:
        form = LoginForm()
    return render(request, 'core/loginpage.html', {'form': form})

def signup_view(request): 
    """
    Handles user signup using the SimpleUserCreationForm.
    """
    if request.method == 'POST':
        form = SimpleUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard_page')
    else:
        form = SimpleUserCreationForm()
    
    return render(request, 'core/signuppage.html', {'form': form})

@login_required(login_url="login_view") 
def logout_view(request):
    """
    Logs the user out.
    """
    logout(request)
    return redirect('homepage')

@login_required(login_url="login_view")
def dashboard_view(request):
    """
    Renders the dashboard page with scan statistics for different plants.
    """
    all_plants = Plant.objects.all().order_by('name')

    # Create a dictionary to hold data for each plant, keyed by plant name
    plants_data = {}

    for plant in all_plants:
        # Group scan history by disease name and count them for the current user and plant
        disease_counts = Scan.objects.filter(
            user=request.user, 
            plant=plant
        ).values('disease__name').annotate(
            count=Count('disease__name')
        ).order_by('-count')  # Order by count descending
        
        chart_labels = []
        chart_data = []
        chart_colors = []
        
        total_scans_for_plant = sum(entry['count'] for entry in disease_counts)

        if total_scans_for_plant > 0:
            for entry in disease_counts:
                disease_name = entry['disease__name']
                count = entry['count']
                
                if disease_name:
                    chart_labels.append(disease_name)
                else:
                    chart_labels.append(str(_("Healthy / No Disease Detected")))
                
                chart_data.append(count)
                
                # Assign colors based on disease type
                if not disease_name or disease_name == "Healthy / No Disease Detected":
                    chart_colors.append('#4CAF50')  # Green for healthy
                elif 'blight' in disease_name.lower():
                    chart_colors.append('#FF5722')  # Red for blight
                elif 'mildew' in disease_name.lower():
                    chart_colors.append('#FFC107')  # Yellow for mildew
                elif 'spot' in disease_name.lower():
                    chart_colors.append('#9C27B0')  # Purple for spots
                else:
                    chart_colors.append('#2196F3')  # Blue for others
        else:
            chart_labels.append(str(_("No Scans Yet")))
            chart_data.append(1)
            chart_colors.append('#9E9E9E')  # Gray for no data

        # Populate the dictionary with the processed data
        plants_data[plant.name] = {
            'labels': chart_labels,
            'data': chart_data,
            'colors': chart_colors,
            'total_scans': total_scans_for_plant,
        }

    context = {
        'user': request.user,
        'plants_data_json': json.dumps(plants_data),
    }
    return render(request, 'core/dashboard page.html', context)

@login_required(login_url="login_view")
def list_page_view(request):
    """
    Renders the list page with plant and disease information.
    """
    current_language = get_language()
    print(f"🌐 Loading list page in language: {current_language}")
    
    plants = Plant.objects.prefetch_related('diseases').all()
    
    plants_with_diseases = []
    for plant in plants:
        # Get translated plant name
        if current_language == 'tl':
            plant_name = plant.get_translated_name('tl') or plant.name
            plant_description = plant.get_translated_description('tl') or plant.description or str(_("No description available for this plant."))
        else:
            plant_name = plant.name
            plant_description = plant.description or str(_("No description available for this plant."))
        
        diseases_data = []
        for disease in plant.diseases.all():
            # Get image URL if exists
            image_url = disease.image.url if disease.image else None
            
            # Get translated disease information based on current language
            if current_language == 'tl':
                disease_data = {
                    'id': disease.id,
                    'name': disease.get_translated_name('tl') or disease.name,
                    'description': disease.get_translated_description('tl') or disease.description or '',
                    'symptoms': disease.get_translated_symptoms('tl') or disease.symptoms or '',
                    'prevention': disease.get_translated_prevention('tl') or disease.prevention or '',
                    'original_name': disease.name,
                    'image_url': image_url,
                }
            else:
                disease_data = {
                    'id': disease.id,
                    'name': disease.name,
                    'description': disease.description or '',
                    'symptoms': disease.symptoms or '',
                    'prevention': disease.prevention or '',
                    'original_name': disease.name,
                    'image_url': image_url,
                }
            diseases_data.append(disease_data)
        
        plants_with_diseases.append({
            'name': plant_name,
            'description': plant_description,
            'diseases': diseases_data
        })

    context = {
        'plants_with_diseases': plants_with_diseases,
        'current_language': current_language,
    }
    
    print(f"📋 Loaded {len(plants_with_diseases)} plants with diseases for language: {current_language}")
    return render(request, 'core/list_page.html', context)

@login_required
def scan_history_api(request):
    """API endpoint to fetch scan history for a specific plant with filtering"""
    plant_name = request.GET.get('plant', '')
    filter_type = request.GET.get('filter', 'all')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    print(f"🔍 API CALL: Fetching scan history for plant: '{plant_name}', Filter: '{filter_type}', User: {request.user.username}")
    
    if not plant_name:
        print("❌ API ERROR: No plant name provided")
        return JsonResponse({'error': 'Plant name required'}, status=400)
    
    try:
        # Base queryset - get all scans for the user and plant
        scans = Scan.objects.filter(
            user=request.user,
            plant__name=plant_name
        ).select_related('plant', 'disease').order_by('-scan_date')
        
        print(f"📊 Initial scan count for {plant_name}: {scans.count()}")
        
        # Apply time-based filtering
        now = timezone.now()
        if filter_type == 'today':
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            scans = scans.filter(scan_date__gte=start_of_day)
            print(f"🕒 Filtered to today: {scans.count()} scans")
            
        elif filter_type == 'week':
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            scans = scans.filter(scan_date__gte=start_of_week)
            print(f"🕒 Filtered to this week: {scans.count()} scans")
            
        elif filter_type == 'month':
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            scans = scans.filter(scan_date__gte=start_of_month)
            print(f"🕒 Filtered to this month: {scans.count()} scans")
            
        elif filter_type == 'custom' and start_date and end_date:
            try:
                # Parse dates from string format (YYYY-MM-DD)
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                
                # Make end_date inclusive (end of the day)
                end_date_obj = end_date_obj.replace(hour=23, minute=59, second=59)
                
                # Convert to timezone-aware datetime if needed
                if timezone.is_naive(start_date_obj):
                    start_date_obj = timezone.make_aware(start_date_obj)
                if timezone.is_naive(end_date_obj):
                    end_date_obj = timezone.make_aware(end_date_obj)
                
                scans = scans.filter(scan_date__range=[start_date_obj, end_date_obj])
                print(f"🕒 Filtered to custom range {start_date} to {end_date}: {scans.count()} scans")
                
            except ValueError as e:
                print(f"❌ Date parsing error: {e}")
                return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)
        
        # Prepare scan data for response
        scan_data = []
        for scan in scans:
            # Get translated disease name if available
            disease_name = None
            if scan.disease:
                current_language = get_language()
                if current_language == 'tl':
                    disease_name = scan.disease.get_translated_name('tl') or scan.disease.name
                else:
                    disease_name = scan.disease.name
            else:
                # Handle case where no disease was detected
                disease_name = _("Healthy / No Disease Detected")
            
            scan_data.append({
                'scan_date': scan.scan_date.isoformat(),
                'plant_type': scan.plant.name,
                'disease_name': disease_name,
                'confidence': float(scan.confidence) if scan.confidence else None,
                'detected_result': scan.detected_result or "Unknown",
            })
        
        print(f"✅ API SUCCESS: Returning {len(scan_data)} scans for plant: {plant_name}")
        
        return JsonResponse({
            'plant': plant_name,
            'filter': filter_type,
            'scans': scan_data,
            'total_count': len(scan_data)
        })
        
    except Exception as e:
        print(f"❌ API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)