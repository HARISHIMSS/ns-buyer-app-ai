from helpers.mongo import getDataFromDB
import pandas as pd

data = getDataFromDB()
keys,values = data["keys"],data["values"]

def getFlattenedDF():
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