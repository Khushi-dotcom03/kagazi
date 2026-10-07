from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(sentences, query):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

    embeddings_query = model.encode(query)
    print(embeddings_query.shape)

    cs = cosine_similarity(embeddings_query, embeddings)
    print(cs.shape)

    best_index = similarities.argmax()

    print("Query:")
    print(query)

    print("\nMost Similar:")
    print(sentences[best_index])




sentences = [
    "Artificial Intelligence helps automate repetitive tasks.",
    "Large Language Models can generate human-like text.",
    "Automobiles are used for transportation.",
    "Cars are fast.",
    "Pizza is a popular fast food.",
    "Fresh fruits are rich in vitamins.",
    "The batsman scored a century in the match.",
    "The bowler took five wickets.",
    "The movie received excellent reviews.",
    "Watching movies is a popular form of entertainment.",
    "College students attend lectures every day.",
    "Assignments help students improve their understanding.",
    "A passport is required for international travel.",
    "An invoice contains payment details."
]

query = [
    "How do vehicles help us travel?"
]

compute_similarity(sentences, query)




