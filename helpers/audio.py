from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os
from tempfile import NamedTemporaryFile
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
def transcribe_audio(audio_file):

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    # Save the uploaded audio file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.file.read())
        temp_audio_path = temp_audio.name

    try:
        # Transcribe audio using the specified model
        result = pipe(temp_audio_path)
        transcription = result["text"]
        return transcription
    finally:
        # Remove the temporary audio file
        os.remove(temp_audio_path)