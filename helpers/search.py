import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic

from helpers.utils.data import getFlattenedDF

df = getFlattenedDF()
# Create an NLP pipeline for extracting embeddings
nlp = pipeline("feature-extraction",model="distilbert-base-cased")

# Embed product names
df["product_embedding"] = df["product_name"].apply(lambda x: np.mean(np.array(nlp(x)[0][:768]), axis=0) if isinstance(x, list) else np.mean(np.array(nlp(x)[0][:768]), axis=0))

# Embed seller names
df["seller_embedding"] = df["seller_name"].apply(lambda x: np.mean(np.array(nlp(x)[0][:768]), axis=0) if isinstance(x, list) else np.mean(np.array(nlp(x)[0][:768]), axis=0))

# Convert the 'product_embedding' and 'seller_embedding' columns to NumPy arrays
product_embedding_array = np.vstack(df["product_embedding"].to_list())
seller_embedding_array = np.vstack(df["seller_embedding"].to_list())

# Vectorize the product and seller names using TF-IDF for a simple full-text search
tfidf_vectorizer_product = TfidfVectorizer(stop_words="english",min_df=1)
tfidf_matrix_product = tfidf_vectorizer_product.fit_transform(df["product_name"])

tfidf_vectorizer_seller = TfidfVectorizer(stop_words="english",min_df=1)
tfidf_matrix_seller = tfidf_vectorizer_seller.fit_transform(df["seller_name"])

# Create nearest neighbors models for TF-IDF search
tfidf_nn_model_product = NearestNeighbors(n_neighbors=5, metric="cosine")
tfidf_nn_model_product.fit(tfidf_matrix_product)

tfidf_nn_model_seller = NearestNeighbors(n_neighbors=5, metric="cosine")
tfidf_nn_model_seller.fit(tfidf_matrix_seller)

# Fit nearest neighbors models for transformer embeddings
transformer_nn_model_product = NearestNeighbors(n_neighbors=5, metric="cosine")
transformer_nn_model_product.fit(product_embedding_array)

transformer_nn_model_seller = NearestNeighbors(n_neighbors=5, metric="cosine")
transformer_nn_model_seller.fit(seller_embedding_array)

# Function for full-text search using TF-IDF for product or seller
def full_text_search_tfidf(query, is_product=True):
    tfidf_vectorizer = tfidf_vectorizer_product if is_product else tfidf_vectorizer_seller
    tfidf_matrix = tfidf_matrix_product if is_product else tfidf_matrix_seller
    nn_model = tfidf_nn_model_product if is_product else tfidf_nn_model_seller
    
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[:-6:-1]
    results = df.iloc[related_indices]
    return results["document"].tolist()

# Function for full-text search using transformer embeddings for product or seller
def full_text_search_transformer(query, is_product=True):
    nn_model = transformer_nn_model_product if is_product else transformer_nn_model_seller
    embedding_array = product_embedding_array if is_product else seller_embedding_array
    
    query_embedding = np.mean(np.array(nlp(query)[0][:768]), axis=0)
    _, indices = nn_model.kneighbors([query_embedding])
    results = df.iloc[indices[0]]
    return results["document"].tolist()

# Function to search by GPS coordinates
def search_by_gps(latitude, longitude):
    query_embedding = np.mean(np.array(nlp(f"{latitude},{longitude}")[0][:768]), axis=0)
    _, indices = transformer_nn_model_product.kneighbors([query_embedding])
    results = df.iloc[indices[0]]
    return results["document"].tolist()

# Function to search for a product by GPS coordinates within a 5 km range
def search_product_by_gps(product_name, latitude, longitude, max_distance_km=100):
    # Get the embedding for the given product name
    product_embedding = np.mean(np.array(nlp(product_name)[0][:768]), axis=0)
    
    # Use the transformer model for product embeddings
    _, indices = transformer_nn_model_product.kneighbors([product_embedding])
    
    # Filter the results based on the GPS coordinates within the specified range
    results = df.iloc[indices[0]]
    results = results[results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, row['gps'].split(',')))).km <= max_distance_km, axis=1)]
    
    return results["document"].tolist()
# Function to search for a seller by GPS coordinates within a 5 km range
def search_seller_by_gps(seller_name, latitude, longitude, max_distance_km=100):
    # Get the embedding for the given product name
    seller_embedding = np.mean(np.array(nlp(seller_name)[0][:768]), axis=0)
    
    # Use the transformer model for product embeddings
    _, indices = transformer_nn_model_product.kneighbors([seller_embedding])
    
    # Filter the results based on the GPS coordinates within the specified range
    results = df.iloc[indices[0]]
    results = results[results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, row['gps'].split(',')))).km <= max_distance_km, axis=1)]
    
    return results["document"].tolist()

def search_by_category(category, latitude, longitude):
    # Filter by category
    category_results = df[df["category"] == category]

    if not category_results.empty:
        # Filter the results based on the GPS coordinates within the specified range
        category_results['gps'] = category_results['gps'].apply(lambda x: tuple(map(float, x.split(','))))
        results = category_results[category_results.apply(lambda row: geodesic((latitude, longitude), row['gps']).km <= 5, axis=1)]

        if not results.empty:
            # Perform full-text search within the filtered results
            tfidf_vectorizer_category = TfidfVectorizer(stop_words="english")
            tfidf_matrix_category = tfidf_vectorizer_category.fit_transform(results["product_name"])

            query_vector_category = tfidf_vectorizer_category.transform([category])
            cosine_similarities_category = linear_kernel(query_vector_category, tfidf_matrix_category).flatten()
            related_indices_category = cosine_similarities_category.argsort()[:-6:-1]
            results = results.iloc[related_indices_category]

            return results["document"].tolist()  # Convert to list for consistent return type
        else:
            return "Items are not found for the 5 km range"  # Return a message if no results are found within the specified range
    else:
        return "Items are not found for the given category"  # Return a message if no results are found for the given category
def search_by_domain(domain, latitude, longitude):
    # Filter by domain
    domain_results = df[df["domain"] == domain]

    if not domain_results.empty:
        # Filter the results based on the GPS coordinates within the specified range
        domain_results['gps'] = domain_results['gps'].apply(lambda x: tuple(map(float, x.split(','))))
        results = domain_results[domain_results.apply(lambda row: geodesic((latitude, longitude), row['gps']).km <= 5, axis=1)]

        if not results.empty:
            # Perform full-text search within the filtered results
            tfidf_vectorizer_domain = TfidfVectorizer(stop_words="english")
            tfidf_matrix_domain = tfidf_vectorizer_domain.fit_transform(results["product_name"])

            query_vector_domain = tfidf_vectorizer_domain.transform([domain])
            cosine_similarities_domain = linear_kernel(query_vector_domain, tfidf_matrix_domain).flatten()
            related_indices_domain = cosine_similarities_domain.argsort()[:-6:-1]
            results = results.iloc[related_indices_domain]

            return results["document"].tolist()  # Convert to list for consistent return type
        else:
            return "Items are not found for the 5 km range"  # Return an empty list if no results are found within the specified range
    else:
        return "Items are not found for the given domain and gps coordinates"  # Return an empty list if no results are found for the given domain

def list_of_available_domains():
    available_domains = df["domain"].unique().tolist()
    return available_domains

def list_of_available_category():
    available_categories = df["category"].unique().tolist()
    return available_categories

def search_products_on_domain_and_gps(domain, product_name, latitude, longitude):
     # Step 1: Search for items in the specified domain within a 5 km range
    domain_results = search_by_domain(domain, latitude, longitude)

    if isinstance(domain_results, list):
        # Step 2: Filter the results based on GPS coordinates within a 5 km range
        results_within_range = [item for item in domain_results if 'gps' in item["locations"] and
                                geodesic((latitude, longitude), tuple(map(float, item["locations"]['gps'].split(',')))).km <= 5]

        print("Results Within Range:", results_within_range)  # Debugging line

        if results_within_range:
            # Step 3: Perform a full-text search within the filtered results for the given product name
            tfidf_vectorizer_product = TfidfVectorizer(stop_words="english")
            tfidf_matrix_product = tfidf_vectorizer_product.fit_transform([item["bpp/providers"]["descriptor"]['name'] for item in results_within_range])

            query_vector_product = tfidf_vectorizer_product.transform([product_name])
            cosine_similarities_product = linear_kernel(query_vector_product, tfidf_matrix_product).flatten()
            related_indices_product = cosine_similarities_product.argsort()[:-6:-1]
            results_final = [results_within_range[i] for i in related_indices_product]

            return results_final
        else:
            return "No items found within the specified GPS coordinates and 5 km range."
    else:
        # Return the message from the search_by_domain function
        return domain_results
    
def search_products_by_domain_category_and_gps(domain, category, product_name, latitude, longitude):
    # Step 1: Filter by domain
    domain_results = df[df["domain"] == domain]

    if not domain_results.empty:
        # Step 2: Filter by category within the domain
        category_results = domain_results[domain_results["category"] == category]

        if not category_results.empty:
            # Step 3: Filter the results based on the GPS coordinates within the specified range
            category_results['gps'] = category_results['gps'].apply(lambda x: tuple(map(float, x.split(','))))
            results = category_results[category_results.apply(lambda row: geodesic((latitude, longitude), row['gps']).km <= 5, axis=1)]

            if not results.empty:
                # Step 4: Perform a full-text search within the filtered results for the given product name
                tfidf_vectorizer_product = TfidfVectorizer(stop_words="english")
                tfidf_matrix_product = tfidf_vectorizer_product.fit_transform(results["product_name"])

                query_vector_product = tfidf_vectorizer_product.transform([product_name])
                cosine_similarities_product = linear_kernel(query_vector_product, tfidf_matrix_product).flatten()
                related_indices_product = cosine_similarities_product.argsort()[:-6:-1]
                results_final = results.iloc[related_indices_product]

                return results_final["document"].tolist()  # Convert to list for consistent return type
            else:
                return "Items are not found for the 5 km range"  # Return a message if no results are found within the specified range
        else:
            return f"No items found for the category '{category}' within the domain '{domain}'."
    else:
        return f"No items found for the given domain '{domain}' and category '{category}'."