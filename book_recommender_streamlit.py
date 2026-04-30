import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Book Recommender")
st.title("📚 Book Recommendation System")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        books = pickle.load(open('books.pkl', 'rb'))
    except Exception as e:
        st.error("❌ Error loading books.pkl file")
        st.stop()

    # Convert to DataFrame if needed
    books = pd.DataFrame(books)

    # 🔥 Normalize column names (VERY IMPORTANT)
    books.columns = books.columns.str.strip().str.lower()

    # 🔍 Debug: Show columns (for YOU to check once)
    st.write("Columns in dataset:", books.columns)

    # 🔥 Detect correct column names dynamically
    title_col = None
    author_col = None
    image_col = None

    for col in books.columns:
        if 'title' in col:
            title_col = col
        if 'author' in col:
            author_col = col
        if 'image' in col or 'url' in col:
            image_col = col

    # ❌ If important columns missing
    if title_col is None or author_col is None:
        st.error("❌ Required columns not found in dataset")
        st.stop()

    # Fill missing image column safely
    if image_col is None:
        books['image'] = "https://via.placeholder.com/150"
        image_col = 'image'

    # 🔥 Create tags
    books['tags'] = books[title_col].astype(str) + " " + books[author_col].astype(str)

    # 🔥 Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(books['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return books, similarity, title_col, image_col

books, similarity, title_col, image_col = load_data()

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(book):
    matches = books[books[title_col].str.contains(book, case=False, na=False)]

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
        names.append(books.iloc[i[0]][title_col])
        images.append(books.iloc[i[0]][image_col])

    return names, images

# -------------------------------
# UI
# -------------------------------
book_list = books[title_col].dropna().unique()
selected_book = st.selectbox("📖 Select a book", book_list)

if st.button("Recommend"):
    names, images = recommend(selected_book)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(images[i])
