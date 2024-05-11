from helpers.mongo import getFlattenedDF
mongo_df = getFlattenedDF()

def updateMongoDF():
    global mongo_df
    mongo_df = getFlattenedDF()
def getMongoDF():
    return mongo_df