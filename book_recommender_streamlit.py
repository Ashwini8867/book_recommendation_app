import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        books = pickle.load(open('books.pkl', 'rb'))
    except Exception:
        st.error("❌ Error loading books.pkl. File missing or empty.")
        st.stop()

    books = pd.DataFrame(books)

    # Normalize column names
    books.columns = books.columns.str.strip().str.lower()

    # -------------------------------
    # Detect columns
    # -------------------------------
    title_col = None
    author_col = None

    for col in books.columns:
        if 'title' in col:
            title_col = col
        if 'author' in col:
            author_col = col

    if title_col is None:
        st.error("❌ No title column found in dataset")
        st.stop()

    # If author missing
    if author_col is None:
        books['author'] = "Unknown"
        author_col = 'author'

    # Fill missing values
    books[title_col] = books[title_col].fillna("")
    books[author_col] = books[author_col].fillna("")

    # -------------------------------
    # Create tags
    # -------------------------------
    books['tags'] = books[title_col] + " " + books[author_col]

    # -------------------------------
    # Vectorization
    # -------------------------------
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(books['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return books, similarity, title_col

books, similarity, title_col = load_data()

# -------------------------------
# Get Book Image (ALWAYS WORKS)
# -------------------------------
def get_book_image(title):
    return f"https://covers.openlibrary.org/b/title/{title.replace(' ', '+')}-M.jpg"

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(book):
    matches = books[books[title_col].str.contains(book, case=False, na=False)]

    if matches.empty:
        return ["No match found"] * 5, ["https://via.placeholder.com/150"] * 5

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
        title = books.iloc[i[0]][title_col]
        names.append(title)
        images.append(get_book_image(title))  # 🔥 always fetch image

    return names, images

# -------------------------------
# UI
# -------------------------------
book_list = books[title_col].dropna().unique()
selected_book = st.selectbox("📖 Select a book", book_list)

# -------------------------------
# Show Recommendations
# -------------------------------
if st.button("Recommend"):
    names, images = recommend(selected_book)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"**{names[i]}**")
            st.image(images[i], use_container_width=True)
