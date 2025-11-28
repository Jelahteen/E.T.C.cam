# detection/models.py

from django.db import models
from django.contrib.auth import get_user_model

# Get the custom User model if you're using one, otherwise default Django User
User = get_user_model()

class Plant(models.Model):
    """
    Represents the type of plant (e.g., Eggplant, Tomato, Chili Pepper).
    This model will store the different plant categories your app handles.
    """
    name = models.CharField(max_length=100, unique=True, help_text="Name of the plant (e.g., Eggplant, Tomato)")
    # ENSURE THIS FIELD IS PRESENT
    description = models.TextField(blank=True, null=True, help_text="Optional description of the plant")
    
    # Filipino translation fields for Plant
    name_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino name")
    description_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino description")
    
    name_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino name correction")
    description_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino description correction")

    def __str__(self):
        return self.name
    
    def get_translated_name(self, language='tl'):
        """Get the best available translation for plant name"""
        if language == 'tl':
            return self.name_manual_tl or self.name_auto_tl or self.name
        return self.name
    
    def get_translated_description(self, language='tl'):
        """Get the best available translation for plant description"""
        if language == 'tl':
            return self.description_manual_tl or self.description_auto_tl or self.description
        return self.description
    
    def auto_translate_all(self):
        """Auto-translate all plant fields to Filipino and save"""
        from .utils import auto_translate_text
        
        if not self.name_auto_tl:
            self.name_auto_tl = auto_translate_text(self.name, 'tl')
        
        if not self.description_auto_tl and self.description:
            self.description_auto_tl = auto_translate_text(self.description, 'tl')
        
        self.save()
        return self

class Disease(models.Model):
    """
    Represents a specific disease that can affect a plant.
    This model will store information about various plant diseases.
    """
    # English fields (original content)
    name = models.CharField(max_length=200, unique=True, help_text="Name of the disease")
    description = models.TextField(blank=True, null=True, help_text="Detailed description of the disease")
    symptoms = models.TextField(blank=True, null=True, help_text="Symptoms of the disease")
    prevention = models.TextField(blank=True, null=True, help_text="Prevention strategies for the disease")
    plants = models.ManyToManyField(Plant, related_name='diseases', blank=True, help_text="Select plants this disease can affect")
    
    # Image field for disease visualization
    image = models.ImageField(
        upload_to='disease_images/',
        blank=True, 
        null=True,
        help_text="Upload an image that represents this disease"
    )
    
    # Auto-translated Filipino fields (system-generated)
    name_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino name")
    description_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino description")
    symptoms_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino symptoms")
    prevention_auto_tl = models.TextField(blank=True, null=True, help_text="Auto-translated Filipino prevention")
    
    # Manual override Filipino fields (for corrections)
    name_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino name correction")
    description_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino description correction")
    symptoms_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino symptoms correction")
    prevention_manual_tl = models.TextField(blank=True, null=True, help_text="Manual Filipino prevention correction")

    def __str__(self):
        return self.name
    
    def get_translated_name(self, language='tl'):
        """Get the best available translation for disease name"""
        if language == 'tl':
            return self.name_manual_tl or self.name_auto_tl or self.name
        return self.name
    
    def get_translated_description(self, language='tl'):
        """Get the best available translation for disease description"""
        if language == 'tl':
            return self.description_manual_tl or self.description_auto_tl or self.description
        return self.description
    
    def get_translated_symptoms(self, language='tl'):
        """Get the best available translation for disease symptoms"""
        if language == 'tl':
            return self.symptoms_manual_tl or self.symptoms_auto_tl or self.symptoms
        return self.symptoms
    
    def get_translated_prevention(self, language='tl'):
        """Get the best available translation for disease prevention"""
        if language == 'tl':
            return self.prevention_manual_tl or self.prevention_auto_tl or self.prevention
        return self.prevention
    
    def auto_translate_all(self):
        """Auto-translate all disease fields to Filipino and save"""
        from .utils import auto_translate_text
        
        # Only translate if auto-translation doesn't exist
        if not self.name_auto_tl:
            self.name_auto_tl = auto_translate_text(self.name, 'tl')
        
        if not self.description_auto_tl and self.description:
            self.description_auto_tl = auto_translate_text(self.description, 'tl')
        
        if not self.symptoms_auto_tl and self.symptoms:
            self.symptoms_auto_tl = auto_translate_text(self.symptoms, 'tl')
        
        if not self.prevention_auto_tl and self.prevention:
            self.prevention_auto_tl = auto_translate_text(self.prevention, 'tl')
        
        self.save()
        return self
    
    def translation_status(self):
        """Helper method to check translation status"""
        if self.name_manual_tl:
            return "✅ Manual"
        elif self.name_auto_tl:
            return "🤖 Auto"
        else:
            return "❌ Not Translated"

class Scan(models.Model):
    """
    Records each instance of a user performing a scan.
    This model links a user to a specific plant and a detected disease, along with a timestamp.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans', help_text="The user who performed the scan")
    plant = models.ForeignKey(Plant, on_delete=models.SET_NULL, null=True, blank=True, help_text="The type of plant scanned (e.g., Eggplant, Tomato)")
    disease = models.ForeignKey(Disease, on_delete=models.SET_NULL, null=True, blank=True, help_text="The disease detected (if any). Can be null if no disease is found.")
    scan_date = models.DateTimeField(auto_now_add=True, help_text="The date and time the scan was performed.")
    scanned_image = models.ImageField(upload_to='scans/', blank=True, null=True)
    detected_result = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        # Orders scans in reverse chronological order (most recent first)
        ordering = ['-scan_date']
        verbose_name = "Scan Record" # Nicer name in Django Admin
        verbose_name_plural = "Scan Records" # Nicer name for plural in Django Admin

    def __str__(self):
        plant_name = self.plant.name if self.plant else 'Unknown Plant'
        disease_name = self.disease.name if self.disease else 'No Disease Detected'
        return f"Scan by {self.user.username} - {plant_name}, Result: {disease_name} on {self.scan_date.strftime('%Y-%m-%d %H:%M')}"
    
    def get_translated_plant_name(self, language='tl'):
        """Get translated plant name for this scan"""
        if self.plant and language == 'tl':
            return self.plant.get_translated_name('tl')
        return self.plant.name if self.plant else 'Unknown Plant'
    
    def get_translated_disease_name(self, language='tl'):
        """Get translated disease name for this scan"""
        if self.disease and language == 'tl':
            return self.disease.get_translated_name('tl')
        return self.disease.name if self.disease else 'No Disease Detected'