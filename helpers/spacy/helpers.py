from helpers.spacy.util import process_query


def search_spacy(text):
     result = process_query(text)
     sellers = [ent.text for ent in result.ents if ent.label_ == "SELLER"]
     products = [ent.text for ent in result.ents if ent.label_ == "PRODUCT"]
     return {"sellers":sellers,"products":products}