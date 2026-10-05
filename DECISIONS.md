# Design decisions

Every choice in this project, the alternative I rejected, and what it costs.

## 1. Tokenizer: delete punctuation instead of splitting on it
- **Chose:** lowercase, then `re.sub(r'[^\w\s]', '', text)`, then split on spaces.
- **Alternative:** extract runs of letters and digits with `re.findall(r'[a-z0-9]+', text)`.
- **Why:** deleting the comma keeps `100,000` as one token (`100000`). For financial news, a number split into `100` and `000` loses its meaning.
- **Cost:** hyphenated words get glued together (`IIT-Jodhpur` → `iitjodhpur`, `don't` → `dont`), so they no longer match their separate parts.

## 2. Vocabulary: rebuild it per query (docs + this query)
- **Chose:** `bag_of_words(DOCS + [query])` on every search.
- **Alternative:** build the vocabulary once from all docs and a fixed list of queries.
- **Why:** real users type queries I have never seen. With a fixed vocabulary, a new word such as "TCS" is missing, so the code either crashes with a `KeyError` or silently ignores the word. Rebuilding with the query included works for any query.
- **Cost:** every document is re-counted on every search. Instant for 12 docs, far too slow for a million. At scale: build the document matrix once and drop unknown query words.

## 3. Similarity: cosine, not a raw dot product
- **Chose:** `(D @ q) / (|q| * |D|)`.
- **Alternative:** the plain dot product `D @ q`.
- **Why:** the dot product grows with text length, so long documents win against every query. Dividing by the lengths compares the *mix* of words (the angle), not the amount.
- **Known gap:** a query with no tokens has length 0, which divides by zero. Not handled yet; the fix is to return all-zero scores when `|q| == 0`.

## 4. Embedding model: all-MiniLM-L6-v2
- **Chose:** `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 384-dimensional vectors).
- **Alternatives:** larger models (e.g. 768 or 1024 dimensions) or a paid embedding API.
- **Why:** small (about 90 MB), fast, fits easily on my 4 GB RTX 3050, free, runs offline, and is a standard baseline that others can compare against.
- **Cost:** a bigger model would likely understand finance terms and aliases better. Not measured yet.

## 5. Encode documents once, before any search
- **Chose:** `doc_vecs = model.encode(DOCS)` at start-up; only the query is encoded per search.
- **Alternative:** re-encode everything per query, the way bag of words rebuilds its matrix.
- **Why:** embeddings need no shared vocabulary, since every text maps to the same 384 numbers. So document vectors can be computed once and reused. This is what a vector database stores.
- **Cost:** when a document changes, its vector must be recomputed.

## 6. Reuse one cosine function for both methods
- **Chose:** the same `cosine(q, D)` scores both bag-of-words vectors and embeddings.
- **Why:** only the "text → numbers" step differs between the two methods, so the comparison is fair and the scoring code is tested once.

## 7. Known weaknesses (found by testing, not assumed)

| Query | Bag of words | Embeddings |
|---|---|---|
| TCS Q2 results | Right top hit; Tata Consultancy headline scores 0.000 | Right top hit; Tata Consultancy headline only 0.285 |
| RBI does not change interest rates | **Wrong:** "hikes" ranks first (0.246 vs 0.167) | Right, but only by 0.018 over "hikes" |
| RBI raises interest rates | Right | Right |
| Bitcoin price crash | Right; #2 is "crosses 100,000" | Right; #2 is "rally, all time high" |

- **Aliases:** neither method knows TCS = Tata Consultancy Services.
- **Negation and direction:** embeddings encode the topic (RBI, rates, Bitcoin), not what happened (up, down, unchanged). Opposite news scores almost as high. In a trading system this is dangerous.
- **Morphology:** to bag of words, "unchanged" and "change" are unrelated tokens. Stemming would help, at the cost of sometimes merging words with different meanings.

**Next steps these point to:** hybrid search (exact names from bag of words + meaning from embeddings), a reranker that reads query and document together, and a separate direction/sentiment signal.
