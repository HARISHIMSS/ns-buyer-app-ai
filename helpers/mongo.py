from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "nsbuyerapp"
COLLECTION_NAME = "ondc_catalog_v2"

# Connect to the MongoDB server
client = MongoClient(MONGO_URI)

# Select the database
db = client[DATABASE_NAME]

# Select the collection
collection = db[COLLECTION_NAME]
def getDataFromDB():
    print("From Mongodb")
    # Query for documents (e.g., retrieve all documents)
    cursor = collection.find({})

    documents = []

    step1_list = []
    # # Iterate over the cursor to access the documents
    for document in cursor:
        # bap_uri= document["bap_uri"]
        # bap_id = document["bap_id"]
        # domain = document["_id"].split("_")[1]
        bap_uri,domain,timestamp = document["_id"].split("_")
        bap_id = bap_uri
        if "result" in document:
            data_result = document["result"]
            if data_result is not None:
                for ind_document in data_result:
                    if ind_document is not None:
                        if "bpp/providers" in ind_document and "bpp/descriptor" in ind_document and "bpp/fulfillments":
                            bpp_providers_document = ind_document["bpp/providers"]
                            bpp_descriptor_document = ind_document["bpp/descriptor"]
                            bpp_fulfillments_document = ind_document["bpp/fulfillments"]
                            if( bpp_providers_document is not None and bpp_descriptor_document is not None and bpp_fulfillments_document is not None):
                                for provider in bpp_providers_document:
                                    if provider is not None:
                                        d = {
                                            "domain":domain,
                                            "bap_id":bap_id,
                                            "bap_uri":bap_uri,
                                            "bpp/descriptor":bpp_descriptor_document,
                                            "bpp/fulfillments":bpp_fulfillments_document,
                                            "bpp/providers":provider
                                        }
                                        step1_list.append(d)
    step2_list = []
    if(len(step1_list)>0):
        for document in step1_list:
            temp_document = document
            provider = document["bpp/providers"]
            for item in provider["items"]:
                temp_document["bpp/providers"]["items"] = item
                step2_list.append(temp_document)

    for document in step2_list:
        temp_document = document
        for loc in document["bpp/providers"]["locations"]:
            temp_document["locations"] = loc
            documents.append(temp_document)


    # # Iterate over the cursor to access the documents
    # for document in cursor:
    #     documents.append(document)
    return documents

    # Close the MongoDB connection
    # client.close()
