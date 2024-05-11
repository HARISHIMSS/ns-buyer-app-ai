from imports import asynccontextmanager,File,FastAPI,Form,Optional,UploadFile
from helpers import initialize_spacy,full_text_search_tfidf,search_spacy,transcribe_audio

@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_spacy()
    yield

app = FastAPI(
    title="Buyer App AI APIs",
    summary="Buyer app AI APIs",
    version="0.0.1",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message":"NS-BUYER-APP-AI server is up and running..."}

@app.post("/audio_search")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...),latitude: float = Form(...), longitude: float = Form(...), max_distance: Optional[float] = Form(5)):
    transcribed_text = transcribe_audio(audio_file)
    print("transcribed_text",transcribed_text)
    if(transcribed_text):
        results = full_text_search_tfidf(transcribed_text,latitude,longitude,max_distance)
        return {"message":results}
    return {"transcribed_text":transcribed_text}
@app.post("/search")
async def search(text:str,latitude:str,longitude:str,max_distance:int = 5):
    results = full_text_search_tfidf(text,latitude,longitude,max_distance)
    return {"message":results}

@app.get("/searchSpacy/{text}")
async def search(text:str):
    results = search_spacy(text)
    return {"message":results}














# @app.post("/searchText")
# async def searchText(text:str,language_code:str = "en"):
#     result = translate_and_correct(text,language_code)
#     print("Translated and Corrected Output:", result)
#     return {"refined Text":result}

# @app.post("/searchByDomain")
# async def searchByDomain(domain:str,latitude:str,longitude:str):
#     results = search_by_domain(domain,latitude,longitude)
#     return {"message":results}

# @app.post("/searchByCategory")
# async def searchByCategory(category:str,latitude:str,longitude:str):
#     results = search_by_category(category,latitude,longitude)
#     return {"message":results}
# @app.get("/searchInTFIDF/{text}")
# async def searchInTFIDF(text:str):
#     results = full_text_search_tfidf(text)
#     return {"message":results}
# @app.get("/availableDomains")
# async def availableDomains():
#     results = list_of_available_domains()
#     return {"message":results}

# @app.get("/availableCategories")
# async def availableCategories():
#     results = list_of_available_category()
#     return {"message":results}

# @app.post("/searchProductsOnDomainAndGPS")
# async def searchProductsOnDomainAndGPS(domain:str,product_name:str,latitude:str,longitude:str):
#     results = search_products_on_domain_and_gps(domain,product_name,latitude,longitude)
#     return {"message":results}

# @app.post("/searchProductsOnDomainAndCategoryAndGPS")
# async def searchProductsOnDomainAndCategoryAndGPS(domain:str,category:str,product_name:str,latitude:str,longitude:str):
#     results = search_products_by_domain_category_and_gps(domain,category,product_name,latitude,longitude)
#     return {"message":results}
# @app.get("/searchSpacy/{text}")
# async def searchSpacy(text:str):
#     results = search_spacy(text)
#     return {"message":results}