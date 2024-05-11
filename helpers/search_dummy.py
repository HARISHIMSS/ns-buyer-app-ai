from imports import np,torch,geodesic,NearestNeighbors
from transformers import DistilBertTokenizer, DistilBertModel
from dependencies import getMongoDF

df = getMongoDF()

# Instantiate the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Create an NLP pipeline for extracting embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return torch.mean(last_hidden_states, dim=1).detach().numpy()

# Embed product names
df["product_embedding"] = df["combined_item_name"].apply(lambda x: get_embedding(x) if isinstance(x, str) else np.zeros((1,768)))

# Embed seller names
df["seller_embedding"] = df["provider_name"].apply(lambda x: get_embedding(x) if isinstance(x, str) else np.zeros((1,768)))

# Convert the 'product_embedding' and 'seller_embedding' columns to NumPy arrays
product_embedding_array = np.vstack(df["product_embedding"].to_numpy())
seller_embedding_array = np.vstack(df["seller_embedding"].to_numpy())

# Create nearest neighbors models for transformer embeddings
transformer_nn_model_product = NearestNeighbors(n_neighbors=5, metric="cosine")
transformer_nn_model_product.fit(product_embedding_array)

transformer_nn_model_seller = NearestNeighbors(n_neighbors=5, metric="cosine")
transformer_nn_model_seller.fit(seller_embedding_array)

# Function for full-text search using transformer embeddings for product or seller
def full_text_search_transformer(query, is_product=True):
    nn_model = transformer_nn_model_product if is_product else transformer_nn_model_seller
    embedding_array = product_embedding_array if is_product else seller_embedding_array
    
    query_embedding = get_embedding(query)
    _, indices = nn_model.kneighbors(query_embedding)
    results = df.iloc[indices[0]]
    return results["document"].tolist()

# Function to search by GPS coordinates
def search_by_gps(latitude, longitude):
    query_embedding = get_embedding(f"{latitude},{longitude}")
    _, indices = transformer_nn_model_product.kneighbors(query_embedding)
    results = df.iloc[indices[0]]
    return results["document"].tolist()

# Function to search for a product by GPS coordinates within a 5 km range
def search_product_by_gps(item_name, latitude, longitude, max_distance_km=5):
    # Get the embedding for the given product name
    product_embedding = get_embedding(item_name)
    
    # Use the transformer model for product embeddings
    _, indices = transformer_nn_model_product.kneighbors(product_embedding)
    
    # Filter the results based on the GPS coordinates within the specified range
    results = df.iloc[indices[0]]
    results = results[results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, row['provider_location'].split(',')))).km <= max_distance_km, axis=1)]
    
    return results["document"].tolist()

# Function to search for a seller by GPS coordinates within a 5 km range
def search_seller_by_gps(provider_name, latitude, longitude, max_distance_km=5):
    # Get the embedding for the given product name
    seller_embedding = get_embedding(provider_name)
    
    # Use the transformer model for product embeddings
    _, indices = transformer_nn_model_product.kneighbors(seller_embedding)
    
    # Filter the results based on the GPS coordinates within the specified range
    results = df.iloc[indices[0]]
    results = results[results.apply(lambda row: geodesic((latitude, longitude), tuple(map(float, row['provider_location'].split(',')))).km <= max_distance_km, axis=1)]
    
    return results["document"].tolist()