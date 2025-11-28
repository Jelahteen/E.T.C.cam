# detection/utils.py
from googletrans import Translator
import logging

logger = logging.getLogger(__name__)

def auto_translate_text(text, dest_lang='tl'):
    """
    Automatically translate text using Google Translate API
    """
    try:
        # Don't translate if text is too short, empty, or already in target language
        if not text or len(text.strip()) < 2:
            return text
            
        translator = Translator()
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        logger.error(f"Translation failed for '{text}': {e}")
        return text  # Return original text if translation fails

def get_best_translation(obj, field_name, language='tl'):
    """
    Helper function to get the best translation for any model field
    Works for both Plant and Disease models
    """
    if language != 'tl':
        return getattr(obj, field_name)
    
    manual_field = f"{field_name}_manual_tl"
    auto_field = f"{field_name}_auto_tl"
    
    # Priority: manual > auto > original
    manual_translation = getattr(obj, manual_field)
    if manual_translation:
        return manual_translation
    
    auto_translation = getattr(obj, auto_field)
    if auto_translation:
        return auto_translation
    
    return getattr(obj, field_name)