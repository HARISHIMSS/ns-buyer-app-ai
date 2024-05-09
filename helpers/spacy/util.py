import spacy
import re
from helpers.text import translate_and_correct
from helpers.utils.data import getFlattenedDF
from spacy.matcher import Matcher
from spacy.tokens import Span

df = getFlattenedDF()

# Load a spaCy language model
nnlp = spacy.load("en_core_web_sm")

# Create a Matcher
matcher = Matcher(nnlp.vocab)

def initialize_spacy():
    print("Initializing Spacy...")

   # Define a pattern for each seller
    for seller in df["provider_name"].unique().tolist():
        if(bool(re.search(r"\s",seller))):
            lower_pattern = [{"LOWER": word.lower()} for word in seller.split()]
            matcher.add("SELLER", [lower_pattern])
        else:
            lower_pattern = [{"LOWER": seller.lower()}]
            matcher.add("SELLER", [lower_pattern])
   # Define a pattern for each product
    for product in df["item_name"].unique().tolist():
        if(bool(re.search(r"\s",product))):
            lower_pattern = [{"LOWER": word.lower()} for word in product.split()]
            matcher.add("PRODUCT", [lower_pattern])
        else:
            lower_pattern = [{"LOWER": product.lower()}]
            matcher.add("PRODUCT", [lower_pattern])

# Function to process user query
def process_query(text):
    text = translate_and_correct(text)
    doc = nnlp(text.lower())
    matches = matcher(doc)
    entities = list(doc.ents)
    
    for match_id, start, end in matches:
        label = nnlp.vocab.strings[match_id]
        span = Span(doc, start, end, label=label)
        entities.append(span)

    doc.ents = spacy.util.filter_spans(entities)
    return doc