from dependencies import spell,getMongoDF
def trainSpellChecker():
    print("trainSpellChecker")
    df = getMongoDF()
    for itemName,providerName in zip(df["combined_item_name"].unique().tolist(),df["provider_name"].unique().tolist()):
        itemNameArray = itemName.split()
        providerNameArray = providerName.split()
        spell.word_frequency.load_words(itemNameArray)
        spell.known(itemNameArray)
        spell.word_frequency.load_words(providerNameArray)
        spell.known(providerNameArray)