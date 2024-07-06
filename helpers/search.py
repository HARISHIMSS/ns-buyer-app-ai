from imports import TfidfVectorizer,NearestNeighbors,linear_kernel,geodesic,pd

from helpers.utils import search_spacy
from dependencies import getMongoDF

df = getMongoDF()

# Create TF-IDF vectorizer for product names
product_vectorizer = TfidfVectorizer(min_df=1)
product_embeddings = product_vectorizer.fit_transform(df["combined_item_name"])

# Create TF-IDF vectorizer for seller names
seller_vectorizer = TfidfVectorizer(min_df=1)
seller_embeddings = seller_vectorizer.fit_transform(df["provider_name"])

# Create nearest neighbors models for TF-IDF embeddings
tfidf_nn_model_product = NearestNeighbors(n_neighbors=None, metric="cosine")
tfidf_nn_model_product.fit(product_embeddings)

tfidf_nn_model_seller = NearestNeighbors(n_neighbors=None, metric="cosine")
tfidf_nn_model_seller.fit(seller_embeddings)

def full_text_search_tfidf(query,latitude,longitude,max_distance_km,input_language:str="en"):
    spacyResults = search_spacy(query,input_language)
    print("spacyResults",spacyResults)
    productsArray = spacyResults["items"]
    providersArray = spacyResults["providers"]
    combined_results = pd.DataFrame()
    if len(productsArray) > 0:
        # Perform full text search for product names
        for product_query in productsArray:
            query_vector = product_vectorizer.transform([product_query])
            cosine_similarities = linear_kernel(query_vector, product_embeddings).flatten()
            related_indices = cosine_similarities.argsort()[:-100:-1]
            # results = df.iloc[related_indices]
            # results = results[results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, [row['provider_location'][1],row['provider_location'][0]]))).km <= max_distance_km, axis=1)]
            # results = results["document"].tolist()
            # combined_results.extend(results)
            results = df.iloc[related_indices].copy()  # Copy the DataFrame to avoid modifying the original one
            results['distance'] = results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, [row['provider_location'][1], row['provider_location'][0]]))).km, axis=1)
            results = results[results['distance'] <= max_distance_km]
            combined_results = pd.concat([combined_results, results]) 
    if len(providersArray) > 0:
        # Perform full text search for seller names
        for seller_query in providersArray:
            query_vector = seller_vectorizer.transform([seller_query])
            cosine_similarities = linear_kernel(query_vector, seller_embeddings).flatten()
            related_indices = cosine_similarities.argsort()[:-100:-1]
            results = df.iloc[related_indices].copy()  # Copy the DataFrame to avoid modifying the original one
            results['distance'] = results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, [row['provider_location'][1], row['provider_location'][0]]))).km, axis=1)
            results = results[results['distance'] <= max_distance_km]
            combined_results = pd.concat([combined_results, results]) 
    # Remove duplicates and group by _id
    if not combined_results.empty:
        print("ifffffff")
        combined_results = combined_results.drop_duplicates(subset='_id').groupby('domain').apply(lambda x: x.to_dict('records')).to_dict()
        return combined_results
    else:
        return {"message":"No Data Found"}
