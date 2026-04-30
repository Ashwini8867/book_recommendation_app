import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

# -------------------------------
# Load Data (with caching)
# -------------------------------
@st.cache_data
def load_data():
    books = pickle.load(open('books.pkl', 'rb'))

    # 🔥 Normalize column names (important)
    books.columns = books.columns.str.strip().str.lower()

    # 🔥 Create tags (combine title + author)
    books['tags'] = books['book-title'] + " " + books['book-author']

    # 🔥 Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(books['tags']).toarray()

    # 🔥 Similarity
    similarity = cosine_similarity(vectors)

    return books, similarity

books, similarity = load_data()

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(book):
    matches = books[books['book-title'].str.contains(book, case=False)]

    if matches.empty:
        return ["No match found"] * 5, [""] * 5

    index = matches.index[0]
    distances = similarity[index]

    book_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    images = []

    for i in book_list:
        names.append(books.iloc[i[0]]['book-title'])
        images.append(books.iloc[i[0]]['image-url-m'])

    return names, images

# -------------------------------
# UI Components
# -------------------------------
book_list = books['book-title'].values
selected_book = st.selectbox("📖 Select a book", book_list)

# -------------------------------
# Button Action
# -------------------------------
if st.button("Recommend"):
    names, images = recommend(selected_book)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.text(names[i])
            if images[i]:
                st.image(images[i])
