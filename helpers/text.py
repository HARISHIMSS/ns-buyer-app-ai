from dependencies import translation_tokenizer,translation_model,spell,torch_device

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
        # Translate input text
        input_text = f"translate {detected_language} to en: {input_text}"
        input_ids = translation_tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        # Generate translation
        outputs = translation_model.generate(input_ids)
        translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Correct spelling using spellchecker
        corrected_text = correct_spelling(translated_text)

        return corrected_text 
    else:
        print("Language detection failed.")
        return None
