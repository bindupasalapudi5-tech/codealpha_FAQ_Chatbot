import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Python?": "Python is a programming language.",
    "What is Machine Learning?": "Machine Learning is a subset of AI.",
    "What is Data Science?": "Data Science is the study of data.",
    "What is NLP?": "NLP stands for Natural Language Processing."
}

questions = list(faqs.keys())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

st.title("FAQ Chatbot")

user_input = st.text_input("Ask a question")

if user_input:
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    st.write("Bot:", faqs[questions[best_match]])
