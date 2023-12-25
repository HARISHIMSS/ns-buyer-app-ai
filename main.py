from fastapi import FastAPI, File, UploadFile
from helpers.audio import transcribe_audio

from helpers.text import translate_and_correct

app = FastAPI(
    title="Buyer App AI APIs",
    summary="Buyer app AI APIs",
    version="0.0.1"
)
@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.get("/searchText/{text}")
async def searchText(text):
    result = translate_and_correct(text)

    print("User Input:", text)
    print("Translated and Corrected Output:", result)
    return {"refined Text":result}

@app.post("/transcribe-audio/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    transcribed_text = transcribe_audio(audio_file)
    refined_text = translate_and_correct(transcribed_text)
    return {"transcribed_text":transcribed_text,"refined Text":refined_text}
