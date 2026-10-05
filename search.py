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


mat, vocab = bag_of_words(DOCS + QUERIES)
print(mat.shape)    # should be (11, 82)
print(vocab[:6])    # should be ['100000', '25', '4', 'a', 'after', 'agency']
print(mat[0].sum()) # how many tokens are in the first headline?