from fastapi import FastAPI, File, UploadFile
from helpers.audio import transcribe_audio
from helpers.search import full_text_search_tfidf, full_text_search_transformer, list_of_available_category, list_of_available_domains, search_by_category, search_by_domain, search_by_gps, search_product_by_gps, search_products_by_domain_category_and_gps, search_products_on_domain_and_gps, search_seller_by_gps
from fastapi.responses import JSONResponse

from helpers.text import translate_and_correct

app = FastAPI(
    title="Buyer App AI APIs",
    summary="Buyer app AI APIs",
    version="0.0.1"
)
@app.get("/")
async def root():
    return {"message":"NS-BUYER-APP-AI server is up and running..."}

@app.get("/searchText/{text}")
async def searchText(text:str):
    result = translate_and_correct(text)

    print("User Input:", text)
    print("Translated and Corrected Output:", result)
    return {"refined Text":result}

@app.post("/transcribe-audio/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    transcribed_text = transcribe_audio(audio_file)
    refined_text = translate_and_correct(transcribed_text)
    return {"transcribed_text":transcribed_text,"refined Text":refined_text}

@app.post("/searchByDomain")
async def searchByDomain(domain:str,latitude:str,longitude:str):
    results = search_by_domain(domain,latitude,longitude)
    return {"message":results}

@app.post("/searchByCategory")
async def searchByCategory(category:str,latitude:str,longitude:str):
    results = search_by_category(category,latitude,longitude)
    return {"message":results}

@app.post("/searchProductWithGPS")
async def searchProductWithGPS(product_name:str,latitude:str,longitude:str):
    results = search_product_by_gps(product_name, latitude, longitude)
    return {"message":results}
@app.post("/searchSellerWithGPS")
async def searchSellerWithGPS(seller_name:str,latitude:str,longitude:str):
    results = search_seller_by_gps(seller_name, latitude, longitude)
    return {"message":results}

@app.post("/searchByGPS")
async def searchByGPS(latitude:str,longitude:str):
    results = search_by_gps(latitude, longitude)
    return {"message":results}

@app.get("/searchInTFIDF/{text}")
async def searchInTFIDF(text:str):
    results = full_text_search_tfidf(text)
    return {"message":results}

@app.get("/searchInTransformer/{text}")
async def search(text:str):
    results = full_text_search_transformer(text)
    return {"message":results}
@app.get("/availableDomains")
async def availableDomains():
    results = list_of_available_domains()
    return {"message":results}

@app.get("/availableCategories")
async def availableCategories():
    results = list_of_available_category()
    return {"message":results}

@app.post("/searchProductsOnDomainAndGPS")
async def searchProductsOnDomainAndGPS(domain:str,product_name:str,latitude:str,longitude:str):
    results = search_products_on_domain_and_gps(domain,product_name,latitude,longitude)
    return {"message":results}

@app.post("/searchProductsOnDomainAndCategoryAndGPS")
async def searchProductsOnDomainAndCategoryAndGPS(domain:str,category:str,product_name:str,latitude:str,longitude:str):
    results = search_products_by_domain_category_and_gps(domain,category,product_name,latitude,longitude)
    return {"message":results}