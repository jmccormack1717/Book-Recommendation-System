from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load data
df = pd.read_csv('books.csv')

# Create TF-IDF matrix using 'genre' and 'author'
df['features'] = df['author'] + ' ' + df['genre']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Book title -> index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, top_n=3):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get('title', '')
    recommendations = get_recommendations(title)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
