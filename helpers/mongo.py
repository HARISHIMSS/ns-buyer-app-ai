from imports import MongoClient
# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "bap_ce_db"
COLLECTION_NAME = "catalog"

# Connect to the MongoDB server
client = MongoClient(MONGO_URI)

# Select the database
db = client[DATABASE_NAME]

# Select the collection
collection = db[COLLECTION_NAME]
def getDataFromDB():
    print("From Mongodb")
    filter = {}
    projection = {"_id":0,"summary":1}
    cursor = collection.find(filter,projection)
    summary_data = []
    for doc in cursor:
        summary_values = doc["summary"]
        summary_data.append(summary_values)
    keys = list(doc["summary"].keys())
    return {"keys":keys,"values":summary_data}

    # Close the MongoDB connection
    # client.close()
