from dependencies import updateMongoDF
from helpers.mongo import getFlattenedDF
from helpers.utils.spellUtils import trainSpellChecker
from helpers.utils.spacyUtils import initialize_spacy

def trainData():
    print("trainData")
    updateMongoDF()
    initialize_spacy()
    trainSpellChecker()