from helpers.spacy.util import process_query


def search_spacy(text):
     result = process_query(text)
     providers = [ent.text for ent in result.ents if ent.label_ == "PROVIDER"]
     items = [ent.text for ent in result.ents if ent.label_ == "ITEM"]
     return {"providers":providers,"items":items}