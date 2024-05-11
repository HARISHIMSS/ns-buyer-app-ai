# from .. import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline,torch
from imports import AutoModelForSpeechSeq2Seq,AutoProcessor,torch,T5ForConditionalGeneration,T5Tokenizer
cudaIsAvailable = torch.cuda.is_available()
print("Is cuda available", cudaIsAvailable)
torch_device = "cuda:0" if cudaIsAvailable else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

audio2Text_model_id = "openai/whisper-large-v3"
audio2Text_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    audio2Text_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
audio2Text_model.to(torch_device)
audio2Text_processor = AutoProcessor.from_pretrained(audio2Text_model_id)
audio2Text_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

translation_model_id = "google/flan-t5-base"
translation_model = T5ForConditionalGeneration.from_pretrained(
    translation_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
translation_model.to(torch_device)
translation_tokenizer = T5Tokenizer.from_pretrained(translation_model_id)