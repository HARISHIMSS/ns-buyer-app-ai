# from .. import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline,torch
from imports import AutoModelForSpeechSeq2Seq,AutoProcessor,torch,T5ForConditionalGeneration,T5Tokenizer,spacy,Matcher,SpellChecker,M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.utils import is_torch_sdpa_available
cudaIsAvailable = torch.cuda.is_available()
spdaIsAvailable = is_torch_sdpa_available()
print("Is sdpa available",spdaIsAvailable)
print("Is cuda available", cudaIsAvailable)
torch_device = "cuda:0" if cudaIsAvailable else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

audio2Text_model_id = "openai/whisper-large-v3"
if(spdaIsAvailable):
    audio2Text_model = AutoModelForSpeechSeq2Seq.from_pretrained(audio2Text_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,attn_implementation="sdpa")
else:
    audio2Text_model = AutoModelForSpeechSeq2Seq.from_pretrained(audio2Text_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
audio2Text_model.to(torch_device)
audio2Text_processor = AutoProcessor.from_pretrained(audio2Text_model_id)
audio2Text_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

translation_model_id = "facebook/m2m100_1.2B"
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_id)
# translation_model.to(torch_device)
translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_id)

spacy_model_id = "en_core_web_sm"
spacy_nlp = spacy.load(spacy_model_id)
spacy_matcher = Matcher(spacy_nlp.vocab)

spell = SpellChecker()

