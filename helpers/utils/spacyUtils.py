
from helpers.text import translate_and_correct
from imports import Span,spacy,re
from dependencies import spacy_matcher, spacy_nlp,getMongoDF

# Function to process user query
def process_query(text,input_language:str="en"):
    text = translate_and_correct(text,input_language)
    doc = spacy_nlp(text.lower())
    matches = spacy_matcher(doc)
    entities = list(doc.ents)
    
    for match_id, start, end in matches:
        label = spacy_nlp.vocab.strings[match_id]
        span = Span(doc, start, end, label=label)
        entities.append(span)

    doc.ents = spacy.util.filter_spans(entities)
    return doc

def initialize_spacy():
    df = getMongoDF()
    print("Initializing Spacy...")

   # Define a pattern for each seller
    for seller in df["provider_name"].unique().tolist():
        sellerArray = []
        if(bool(re.search(r"\s",seller))):
            for word in seller.split():
                sellerArray.append([{"LOWER":word.lower()}])
        else:
            lower_pattern = [{"LOWER": seller.lower()}]
            sellerArray.append(lower_pattern)
        spacy_matcher.add("PROVIDER", sellerArray)
   # Define a pattern for each product
    for product in df["combined_item_name"].unique().tolist():
        productArray = []
        if(bool(re.search(r"\s",product))):
            for word in product.split():
                productArray.append([{"LOWER":word.lower()}])
        else:
            lower_pattern = [{"LOWER": product.lower()}]
            productArray.append(lower_pattern)
        spacy_matcher.add("ITEM", productArray)

def search_spacy(text,input_language):
     result = process_query(text,input_language)
     providers = [ent.text for ent in result.ents if ent.label_ == "PROVIDER"]
     items = [ent.text for ent in result.ents if ent.label_ == "ITEM"]
     return {"providers":providers,"items":items}