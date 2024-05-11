from dependencies import audio2Text_model,audio2Text_processor,audio2Text_torch_dtype,torch_device
from imports import pipeline,NamedTemporaryFile,os

pipe = pipeline(
        "automatic-speech-recognition",
        model=audio2Text_model,
        tokenizer=audio2Text_processor.tokenizer,
        feature_extractor=audio2Text_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=audio2Text_torch_dtype,
        device=torch_device
    )
def transcribe_audio(audio_file):
    # Save the uploaded audio file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.file.read())
        temp_audio_path = temp_audio.name

    try:
        # Transcribe audio using the specified model
        result = pipe(temp_audio_path,generate_kwargs={"task":"translate"})
        transcription = result["text"]
        return transcription
    finally:
        # Remove the temporary audio file
        os.remove(temp_audio_path)