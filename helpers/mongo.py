from imports import MongoClient,pd

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


def getFlattenedDF():
    data = getDataFromDB()
    keys,values = data["keys"],data["values"]
    df_data = []
    for obj in values:
        itemName = obj["item_name"]
        itemShortDesc = obj["item_short_desc"]
        itemLongDesc = obj["item_long_desc"]
        combined_item_name = ""
        if itemName is not None:
            combined_item_name += itemName
        if itemShortDesc is not None:
            combined_item_name += " " + itemShortDesc
        if itemLongDesc is not None:
            combined_item_name += " " + itemLongDesc
        new_obj = {"combined_item_name":combined_item_name}
        merged_obj = {**obj, **new_obj}
        df_data.append(merged_obj)
    final_keys = keys.append("document")
    df = pd.DataFrame(df_data,columns=final_keys)
    return df
