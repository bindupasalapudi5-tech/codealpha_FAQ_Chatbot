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

print("FAQ Chatbot")
print("Type 'exit' to quit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()

    print("Bot:", faqs[questions[best_match]])