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
    # Query for documents (e.g., retrieve all documents)
    cursor = collection.find({})

    documents = []
    # # Iterate over the cursor to access the documents
    for document in cursor:
        documents.append(document)
    return documents

    # Close the MongoDB connection
    # client.close()
