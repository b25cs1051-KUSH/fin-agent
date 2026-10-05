import re 
#re is regular expression (regex) module in python which is used to work with regular expressions. It provides functions to search, match, and manipulate strings based on patterns.
import numpy as np

DOCS = [
    "RBI raises the repo rate by 25 basis points to curb inflation",
    "Reserve Bank of India hikes interest rates as prices keep rising",
    "Infosys shares fall 4 percent after weak quarterly guidance",
    "IT stocks slide as Infosys cuts its revenue forecast",
    "Bitcoin crosses 100,000 dollars for the first time",
    "Crypto markets rally as BTC hits a new all time high",
    "ISRO launches a new earth observation satellite from Sriharikota",
    "Indian space agency puts a remote sensing satellite into orbit",
]
QUERIES = [
    "central bank increases borrowing costs",
    "tech company stock drops on poor outlook",
    "satellite launch by India",
]

def tokenize(text):
    """
    Tokenizes the input text into lowercase words, removing punctuation.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    tokens = text.split()
    return tokens


def bag_of_words(texts):
    texts = [tokenize(text) for text in texts]
    unique_words =set()
    for text in texts:
        for token in text:
            unique_words.add(token)
    vocab = sorted(list(unique_words))

    index_map = {word : i for i, word in enumerate(vocab)}

    rows = len(texts)
    cols = len(vocab)
    mat = np.zeros((rows,cols))

    for row,text in enumerate(texts):
        for token in text:
                col = index_map[token]
                mat[row][col] += 1

    return mat, vocab

def cosine(q,D):
     dot_product = np.dot(D,q)
     norm_q = np.linalg.norm(q)
     norm_D = np.linalg.norm(D,axis = 1)
     return dot_product / (norm_q * norm_D)

def search(query, k=2):
     matrix ,vocab = bag_of_words(DOCS + [query])
     D = matrix[:len(DOCS),:]
     q = matrix[len(DOCS),:]
     scores = cosine(q, D)
     top_indices = np.argsort(-scores)[:k]
     return [DOCS[i] for i in top_indices] , scores[top_indices]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2',device ="cuda")
docs_vecs = model.encode(DOCS)

def embed_search(query,k=2):
    q = model.encode(query)
    scores = cosine(q,docs_vecs)
    top_indices = np.argsort(-scores)[:k]
    return [DOCS[i] for i in top_indices],scores[top_indices]

print("\n=== EMBEDDINGS ===")
print("doc_vecs shape:", docs_vecs.shape)

for q in QUERIES:
     print(q)
     docs,scores = embed_search(q)
     for doc, s in zip(docs,scores):
          print(f"  {s:.3f}  {doc}")