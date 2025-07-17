Movie Recommendation System (Content-Based Filtering)
This project implements a movie recommendation system using content-based filtering. The system recommends movies similar to a given input movie based on their textual features, such as genre, keywords, cast, crew (director), and plot overview.

1. Project Overview
The goal of this project is to provide personalized movie recommendations to users. Unlike collaborative filtering, which relies on user behavior, content-based filtering leverages the intrinsic properties of the items themselves. This approach is particularly useful for recommending items to new users (cold-start problem) or for recommending niche items.

2. Model Methodology
Our recommendation model follows these key steps:

Data Acquisition and Merging: We use two datasets from Kaggle:

tmdb_5000_movies.csv: Contains movie details like title, genres, keywords, overview, etc.

tmdb_5000_credits.csv: Contains cast and crew information.
These datasets are merged based on the movie title.

Feature Extraction and Preprocessing:

Relevant textual features (genres, keywords, cast, crew, overview) are extracted.

Stringified lists (e.g., "[ {'id': 28, 'name': 'Action'}, ... ]") are converted into clean lists of names.

Only the top 3 cast members and the director are considered for cast and crew respectively.

Spaces within multi-word tags (e.g., "Science Fiction" to "ScienceFiction") are removed to ensure they are treated as single tokens by the vectorizer.

All processed features are concatenated into a single "tags" string for each movie.

The "tags" are converted to lowercase.

Text Vectorization (TF-IDF):

Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert the textual "tags" into numerical vectors. TF-IDF assigns weights to words based on their frequency in a document and their rarity across all documents, highlighting important terms.

A TfidfVectorizer with max_features=5000 (top 5000 most important words) and stop_words='english' (common English words like "the", "is", "a" are removed) is employed.

Similarity Calculation (Cosine Similarity):

Cosine similarity is calculated between all movie vectors. Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 (opposite) to 1 (identical). A higher cosine similarity indicates greater resemblance between movies.

Recommendation Generation:

Given a movie title, the system finds its corresponding vector.

It then retrieves the cosine similarity scores between this movie and all other movies.

The movies are sorted by their similarity scores in descending order.

The top 10 most similar movies (excluding the input movie itself) are returned as recommendations.

3. Project Structure
The project is organized as follows:

Movie_Recommendation_System/
├── app.py                     # Flask application for the web interface and API
├── model_builder.py           # Script to preprocess data and generate the model artifacts
├── tmdb_5000_movies.csv       # Dataset: Movie details
├── tmdb_5000_credits.csv      # Dataset: Movie cast and crew
├── tfidf_vectorizer.pkl       # Pickled TF-IDF Vectorizer
├── similarity_matrix.pkl      # Pickled Cosine Similarity Matrix
├── movies_df_lite.pkl         # Pickled essential movie data (id and title)
├── movie_indices.pkl          # Pickled mapping of movie titles to their indices
├── requirements.txt           # Python dependencies
└── templates/
    └── index.html             # HTML template for the user interface
