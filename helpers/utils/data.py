from helpers.mongo import getDataFromDB
import pandas as pd

data = getDataFromDB()

def getFlattenedData():
    flattened_data = []
    for document in data:
        bpp_providers_document = document["bpp/providers"]
        bpp_descritor_document = document["bpp/descriptor"]
        bpp_fulfillments_document = document["bpp/fulfillments"]
        descriptor_provider = bpp_providers_document["descriptor"]
    #     locations_provider = bpp_providers_document["locations"]
        locations_provider = document["locations"]
        items_provider = bpp_providers_document["items"]
        product_name = descriptor_provider["name"]
        seller_name = bpp_descritor_document["name"]
        category = items_provider["category_id"]
        domain = document["domain"]
        gps = locations_provider["gps"]
        final_document = {
            "product_name" : product_name,
            "seller_name" : seller_name,
            "category" : category,
            "domain" : domain,
            "gps" : gps,
            "document" : document
        }
        flattened_data.append(final_document)
    return flattened_data
def getFlattenedDF():
    return pd.DataFrame(getFlattenedData())
