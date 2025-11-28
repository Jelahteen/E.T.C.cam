# detection/management/commands/translate_diseases.py
from django.core.management.base import BaseCommand
from detection.models import Disease, Plant
from detection.utils import auto_translate_text

class Command(BaseCommand):
    help = 'Bulk translate all diseases and plants to Filipino'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-translation of already translated diseases',
        )
        parser.add_argument(
            '--plants-only',
            action='store_true',
            help='Translate only plants',
        )
        parser.add_argument(
            '--diseases-only', 
            action='store_true',
            help='Translate only diseases',
        )
    
    def handle(self, *args, **options):
        force = options['force']
        plants_only = options['plants_only']
        diseases_only = options['diseases_only']
        
        # Translate Plants
        if not diseases_only:
            plants = Plant.objects.all()
            self.stdout.write(f"🌿 Translating {plants.count()} plants to Filipino...")
            
            plant_success_count = 0
            for plant in plants:
                # Skip if already has auto-translation and not forcing
                if not force and plant.name_auto_tl:
                    self.stdout.write(f"⏭️  Skipping plant (already translated): {plant.name}")
                    continue
                    
                self.stdout.write(f"🔤 Translating plant: {plant.name}")
                
                try:
                    plant.auto_translate_all()
                    plant_success_count += 1
                    self.stdout.write(f"✅ Translated plant: {plant.name} → {plant.name_auto_tl}")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to translate plant {plant.name}: {e}")
            
            self.stdout.write(f"🌿 Plant translation complete! {plant_success_count}/{plants.count()} plants translated.")
        
        # Translate Diseases
        if not plants_only:
            diseases = Disease.objects.all()
            self.stdout.write(f"🦠 Translating {diseases.count()} diseases to Filipino...")
            
            disease_success_count = 0
            for disease in diseases:
                # Skip if already has auto-translation and not forcing
                if not force and disease.name_auto_tl:
                    self.stdout.write(f"⏭️  Skipping disease (already translated): {disease.name}")
                    continue
                    
                self.stdout.write(f"🔤 Translating disease: {disease.name}")
                
                try:
                    disease.auto_translate_all()
                    disease_success_count += 1
                    self.stdout.write(f"✅ Translated disease: {disease.name} → {disease.name_auto_tl}")
                except Exception as e:
                    self.stdout.write(f"❌ Failed to translate disease {disease.name}: {e}")
            
            self.stdout.write(f"🦠 Disease translation complete! {disease_success_count}/{diseases.count()} diseases translated.")
        
        total_translated = (plant_success_count if not diseases_only else 0) + (disease_success_count if not plants_only else 0)
        self.stdout.write(self.style.SUCCESS(f"🎉 Bulk translation complete! {total_translated} items translated successfully!"))