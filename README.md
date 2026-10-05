# fin-agent

Semantic search over financial news headlines, built from scratch. It is step one of a finance agent that will answer questions about markets using live news.

Two retrieval methods, compared on the same data:
- **Bag of words:** my own tokenizer, word-count vectors and cosine similarity in NumPy.
- **Embeddings:** `all-MiniLM-L6-v2` sentence embeddings on GPU, scored with the same cosine function.

## Run it

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows; on Mac/Linux: source .venv/bin/activate
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build
pip install sentence-transformers
python search.py
```

## Results

Top result per query on 12 headlines:

| Query | Bag of words | Embeddings |
|---|---|---|
| tech company stock drops on poor outlook | ✗ no shared words, all scores 0.000 | ✓ Infosys shares fall... (0.533) |
| central bank increases borrowing costs | ✓ Reserve Bank of India hikes... (0.135) | ✓ Reserve Bank of India hikes... (0.439) |
| satellite launch by India | ✓ ISRO launches... (0.167) | ✓ Indian space agency... (0.680) |
| TCS Q2 results | ✓ TCS reports... (0.365) | ✓ TCS reports... (0.643) |
| RBI does not change interest rates | ✗ picks "hikes" (0.246) | ✓ keeps rate unchanged (0.649), only 0.018 ahead of "hikes" |
| RBI raises interest rates | ✓ (0.302) | ✓ (0.695) |
| Bitcoin price crash | ✓ falls below 50,000 (0.258) | ✓ falls below 50,000 (0.601) |

**Takeaway:** bag of words is good at exact names, embeddings are good at meaning, and neither understands direction (up, down, unchanged). Scores are only comparable within one method, not across methods.

Design choices and trade-offs: see [DECISIONS.md](DECISIONS.md).

## Roadmap
- [x] Bag-of-words search from scratch
- [x] Embedding search on GPU
- [ ] Retrieval evaluation (recall@k on a labelled question set)
- [ ] Hybrid search (bag of words + embeddings)
- [ ] LLM answers with citations
- [ ] Agent loop with tools (news search, price lookup)
- [ ] Live news ingestion, FastAPI, Docker, deployment
