from imports import SpellChecker
from helpers.utils import getFlattenedDF
spell = SpellChecker()
df = getFlattenedDF()
for itemName,providerName in zip(df["combined_item_name"].unique().tolist(),df["provider_name"].unique().tolist()):
    itemNameArray = itemName.split()
    providerNameArray = providerName.split()
    spell.word_frequency.load_words(itemNameArray)
    spell.known(itemNameArray)
    spell.word_frequency.load_words(providerNameArray)
    spell.known(providerNameArray)