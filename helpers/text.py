import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langdetect import detect
from spellchecker import SpellChecker

model_id = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_id)
tokenizer = T5Tokenizer.from_pretrained(model_id)
tokenizer, model

def detect_language(text):
    try:
        language_code = detect(text)
        return language_code
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

def correct_spelling(text):
    spell = SpellChecker()
    corrected_tokens = [spell.correction(token) for token in text.split()]
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def translate_and_correct(input_text):
    detected_language = detect_language(input_text)

    if detected_language:
        translation_tokenizer = tokenizer
        translation_model = model
        
        # Translate input text
        input_text = f"translate {detected_language} to en: {input_text}"
        input_ids = translation_tokenizer(input_text, return_tensors="pt").input_ids

        # Generate translation
        outputs = translation_model.generate(input_ids)
        translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Correct spelling using spellchecker
        corrected_text = correct_spelling(translated_text)

        return corrected_text 
    else:
        print("Language detection failed.")
        return None
