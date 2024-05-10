import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from langdetect import detect
from spellchecker import SpellChecker

from helpers.utils.data import getFlattenedDF

model_id = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_id)
tokenizer = T5Tokenizer.from_pretrained(model_id)
tokenizer, model

spell = SpellChecker()
df = getFlattenedDF()
for itemName,providerName in zip(df["combined_item_name"].unique().tolist(),df["provider_name"].unique().tolist()):
    itemNameArray = itemName.split()
    providerNameArray = providerName.split()
    spell.word_frequency.load_words(itemNameArray)
    spell.known(itemNameArray)
    spell.word_frequency.load_words(providerNameArray)
    spell.known(providerNameArray)

def correct_spelling(text):
    corrected_tokens = [spell.correction(token) for token in text.split()]
    print("corrected_tokens",corrected_tokens)
    if any(element is not None for element in corrected_tokens):
        if(corrected_tokens):
            corrected_text = ' '.join(corrected_tokens)
            return corrected_text
        else:
            return text
    else:
        return text

def translate_and_correct(input_text:str,detected_language:str="en"):

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
