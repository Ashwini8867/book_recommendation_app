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
    except Exception as e:
        st.error("❌ Error loading books.pkl. Make sure file exists and is not empty.")
        st.stop()

    books = pd.DataFrame(books)

    # Normalize column names
    books.columns = books.columns.str.strip().str.lower()

    # -------------------------------
    # Detect columns automatically
    # -------------------------------
    title_col = None
    author_col = None
    image_col = None

    for col in books.columns:
        if 'title' in col:
            title_col = col
        elif 'author' in col:
            author_col = col
        elif 'image' in col or 'img' in col or 'url' in col:
            image_col = col

    # Safety checks
    if title_col is None:
        st.error("❌ No title column found in dataset")
        st.stop()

    if author_col is None:
        books['author'] = "Unknown"
        author_col = 'author'

    # Handle image column
    if image_col is None:
        books['image'] = "https://via.placeholder.com/150"
        image_col = 'image'

    # Fill missing values
    books[title_col] = books[title_col].fillna("")
    books[author_col] = books[author_col].fillna("")
    books[image_col] = books[image_col].fillna("https://via.placeholder.com/150")

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

    return books, similarity, title_col, image_col

books, similarity, title_col, image_col = load_data()

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
        names.append(books.iloc[i[0]][title_col])
        img = books.iloc[i[0]][image_col]

        # Validate image URL
        if isinstance(img, str) and img.startswith("http"):
            images.append(img)
        else:
            images.append("https://via.placeholder.com/150")

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
            st.image(images[i])
