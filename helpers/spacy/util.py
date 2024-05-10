import spacy
import re
from helpers.text import translate_and_correct
from helpers.utils.data import getFlattenedDF
from spacy.matcher import Matcher
from spacy.tokens import Span
import pandas as pd

df = getFlattenedDF()
# Load a spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a Matcher
matcher = Matcher(nlp.vocab)

def initialize_spacy():
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
        matcher.add("PROVIDER", sellerArray)
   # Define a pattern for each product
    for product in df["combined_item_name"].unique().tolist():
        productArray = []
        if(bool(re.search(r"\s",product))):
            for word in product.split():
                productArray.append([{"LOWER":word.lower()}])
        else:
            lower_pattern = [{"LOWER": product.lower()}]
            productArray.append(lower_pattern)
        matcher.add("ITEM", productArray)
        
# Function to process user query
def process_query(text):
    text = translate_and_correct(text)
    doc = nlp(text.lower())
    matches = matcher(doc)
    entities = list(doc.ents)
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = Span(doc, start, end, label=label)
        entities.append(span)

    doc.ents = spacy.util.filter_spans(entities)
    return doc