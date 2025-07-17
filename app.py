import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd # Needed if you unpickle movies_df back into a DataFrame

app = Flask(__name__)

# Load the pickled model components
# Ensure these files are in your Render project directory when deployed
try:
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    similarity = pickle.load(open('similarity_matrix.pkl', 'rb'))
    movies_df_dict = pickle.load(open('movies_df_lite.pkl', 'rb'))
    movie_indices_dict = pickle.load(open('movie_indices.pkl', 'rb'))

    # Reconstruct DataFrame and Series from dictionaries for easier use
    # In a real app, you might want to optimize how you load/store this
    movies_df_loaded = pd.DataFrame(movies_df_dict)
    movie_indices_loaded = pd.Series(movie_indices_dict)

except FileNotFoundError:
    print("Error: Pickle files not found. Make sure they are in the same directory and you've run model_builder.py first.")
    # In a production environment, you might want to log this and serve a 500 error page
    exit() # For demonstration, exit if files aren't there

# Recommendation function (copied from the model_builder.py)
def recommend(movie_title):
    if movie_title not in movie_indices_loaded:
        return [] # Movie not found

    idx = movie_indices_loaded[movie_title]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices_recommend = [i[0] for i in sim_scores]
    return movies_df_loaded['title'].iloc[movie_indices_recommend].tolist()

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    movie_name = request.args.get('movie')
    if not movie_name:
        return jsonify({"error": "Please provide a movie title in the 'movie' query parameter."}), 400

    recommendations = recommend(movie_name)
    if not recommendations:
        return jsonify({"message": f"Could not find recommendations for '{movie_name}'. Check spelling or movie might not be in database."}), 404
    return jsonify({"movie_title": movie_name, "recommendations": recommendations})

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production