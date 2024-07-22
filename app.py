import json
import pandas as pd
from flask import Flask, request
from flask_restful import Resource, Api, marshal_with, fields
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

# Set of stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
api = Api(app)

# Define fields for marshalling
recommendationFields = {
    'product_id': fields.String,
    'score': fields.Float
}

# Global variables to store data
products = []
purchases = []

# Function to clean description
def clean_description_nltk(description):
    soup = BeautifulSoup(description, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

# Function to fetch product data
def fetch_product_data():
    product_data = []
    for product in products:
        cleaned_description = clean_description_nltk(product['description'])
        product_data.append({
            'product_id': str(product['_id']),
            'description': cleaned_description,
            'brand': product['brand'],
            'category': product['category']['name'],
            'subcategory': product['subcategory']['name'],
            'rating': product['rating']
        })
    df = pd.DataFrame(product_data)
    # print("Products DataFrame:\n", df)  # Debug statement
    return df

# Function to fetch purchase data
def fetch_purchase_data():
    purchase_data = []
    for purchase in purchases:
        purchase_data.append({
            'user_id': purchase['user_id'],
            'product_id': str(purchase['_id']),
            'rating': purchase['rating']
        })
    df = pd.DataFrame(purchase_data)
    # print("Purchases DataFrame:\n", df)  # Debug statement
    return df

def get_content_recommendations(product_id, products_df, cosine_sim, top_n=5):
    product_idx_list = products_df[products_df['product_id'] == product_id].index.tolist()
    if not product_idx_list:
        # print(f"Product ID {product_id} not found in products DataFrame")  # Debug statement
        return []
    product_idx = product_idx_list[0]
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return [(products_df['product_id'].iloc[i], sim_scores[j][1]) for j, i in enumerate(product_indices)]

def get_hybrid_recommendations(user_id, top_n=5):
    products_df = fetch_product_data()
    purchases_df = fetch_purchase_data()

    # Content-based filtering using product descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Collaborative filtering using purchase data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(purchases_df[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    
    def get_cf_recommendations(user_id, algo, purchases_df, top_n=5):
        all_product_ids = purchases_df['product_id'].unique()
        user_purchased_products = purchases_df[purchases_df['user_id'] == user_id]['product_id'].unique()
        predictions = [algo.predict(user_id, pid) for pid in all_product_ids if pid not in user_purchased_products]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_predictions = predictions[:top_n]
        return [(pred.iid, pred.est) for pred in top_predictions]

    cf_recommendations = get_cf_recommendations(user_id, algo, purchases_df, top_n)
    # print(f"CF Recommendations for User {user_id}:\n", cf_recommendations)  # Debug statement
    
    cf_recommendation_ids = [rec[0] for rec in cf_recommendations]
    content_recommendations = []
    for product_id, cf_score in cf_recommendations:
        content_recommendations.extend(get_content_recommendations(product_id, products_df, cosine_sim, top_n=2))
    # print(f"Content Recommendations based on CF Recommendations:\n", content_recommendations)  # Debug statement
    
    final_recommendations = list(set(cf_recommendations + content_recommendations))
    final_recommendations = sorted(final_recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    # print(f"Final Recommendations:\n", final_recommendations)  # Debug statement

    return [{'product_id': rec[0], 'score': rec[1]} for rec in final_recommendations]

class Recommendations(Resource):
    @marshal_with(recommendationFields)
    def get(self, user_id):
        recommendations = get_hybrid_recommendations(user_id, top_n=5)
        return recommendations

class Products(Resource):
    def post(self):
        global products
        products = request.json
        return {'message': 'Products data received'}, 200

class Purchases(Resource):
    def post(self):
        global purchases
        purchases = request.json
        return {'message': 'Purchases data received'}, 200

api.add_resource(Recommendations, '/recommendations/<string:user_id>')
api.add_resource(Products, '/products')
api.add_resource(Purchases, '/purchases')

if __name__ == '__main__':
    app.run(debug=True)
