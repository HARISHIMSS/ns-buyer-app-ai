from helpers.mongo import getDataFromDB
import pandas as pd

data = getDataFromDB()
keys,values = data["keys"],data["values"]

def getFlattenedDF():
    df_data = []
    for obj in values:
        new_obj = {"document":obj}
        merged_obj = {**obj, **new_obj}
        df_data.append(merged_obj)
    final_keys = keys.append("document")
    df = pd.DataFrame(df_data,columns=final_keys)
    return df