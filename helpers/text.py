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

def translate_and_correct(input_text:str,input_language:str="en"):

    if input_language:
        # Translate input text
        # input_text = f"translate {detected_language} to en: {input_text}"
        # input_ids = translation_tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        # Generate translation
        # outputs = translation_model.generate(input_ids)
        # translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

        translation_tokenizer.src_lang = input_language
        encoded_hi = translation_tokenizer(input_text, return_tensors="pt")
        generated_tokens = translation_model.generate(**encoded_hi, forced_bos_token_id=translation_tokenizer.get_lang_id("en"))
        translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        print("translated_text",translated_text)
        
        # Correct spelling using spellchecker
        corrected_text = correct_spelling(translated_text[0])

        return corrected_text 
    else:
        print("Language detection failed.")
        return None
