# Corrective-Rag-CRAG

A step-by-step implementation of **Corrective RAG (CRAG)** using LangChain, LangGraph, OpenAI and Tavily web search.
Each notebook adds one piece of the pipeline; `final_CRAG.ipynb` puts it all together.

| Notebook | What it adds |
|----------|--------------|
| `1_basic_rag.ipynb` | Basic RAG: PDF → chunks → FAISS → answer with `gpt-4o-mini` |
| `2_retrieval_refinement.ipynb` | Sentence-level decomposition + LLM filter of retrieved context |
| `3_retrieval_evaluator.ipynb` | LLM scores each retrieved chunk (CORRECT / INCORRECT / AMBIGUOUS) |
| `4_web_search_refinement.ipynb` | Falls back to Tavily web search when retrieval is incorrect |
| `5_query_rewrite.ipynb` | Rewrites the question into a keyword web-search query |
| `6_ambiguous.ipynb` | AMBIGUOUS case: combines internal docs + web results |
| `final_CRAG.ipynb` | The complete CRAG graph |

## Prerequisites

- Python 3.10+ (developed on 3.13)
- An **OpenAI API key** — https://platform.openai.com/api-keys
- A **Tavily API key** (needed from notebook 4 onwards) — https://app.tavily.com

## Setup

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd Corrective-Rag-CRAG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv crag
   ```

   - Windows (PowerShell): `crag\Scripts\Activate.ps1`
   - macOS / Linux: `source crag/bin/activate`

3. **Install dependencies** (plus Jupyter to run the notebooks)

   ```bash
   pip install -r requirements.txt jupyter
   ```

4. **Create a `.env` file** in the project root (same folder as the notebooks):

   ```env
   OPENAI_API_KEY=sk-your-openai-key
   TAVILY_API_KEY=tvly-your-tavily-key
   ```

   | Variable | Used for | Required by |
   |----------|----------|-------------|
   | `OPENAI_API_KEY` | Embeddings (`text-embedding-3-large`) and the LLM (`gpt-4o-mini`) | All notebooks |
   | `TAVILY_API_KEY` | Web search via `TavilySearchResults` | Notebooks 4, 5, 6 and `final_CRAG` |

   The notebooks call `load_dotenv()` to read this file. `.env` is already in `.gitignore` — never commit your keys.

5. **Add the source document.** The notebooks load `./documents/ML Book.pdf`.
   Place your PDF there, or change the path in the `PyPDFLoader(...)` cell to use your own file.

## Usage

```bash
jupyter notebook
```

Open the notebooks in order (`1_` → `6_`) to follow the build-up, or go straight to `final_CRAG.ipynb` and run all cells.

### How the final graph works

```
START → retrieve → eval_each_doc ─┬─ CORRECT ──────────────────────────────→ refine → generate → END
                                  └─ INCORRECT / AMBIGUOUS → rewrite_query → web_search ─┘
```

- **CORRECT**: refines and answers from the internal PDF chunks only.
- **INCORRECT**: rewrites the question into a web query, searches Tavily, and answers from web results only.
- **AMBIGUOUS**: searches the web as well, then combines the relevant PDF chunks with the web results.

### Asking your own question

The last cell (**Run example**) calls the compiled graph `app`. Change the `"question"` value and re-run that cell:

```python
res = app.invoke({
    "question": "Batch normalization vs layer normalization",
    "docs": [], "good_docs": [], "verdict": "", "reason": "",
    "strips": [], "kept_strips": [], "refined_context": "",
    "web_query": "", "web_docs": [], "answer": "",
})

print("VERDICT:", res["verdict"])      # CORRECT / INCORRECT / AMBIGUOUS
print("REASON:", res["reason"])
print("WEB_QUERY:", res["web_query"])  # empty when the CORRECT path was taken
print("\nOUTPUT:\n", res["answer"])
```

The graph-building cell ends with `app`, so Jupyter renders a diagram of the graph.

### Tuning

In `final_CRAG.ipynb`:

- `UPPER_TH = 0.7` — a chunk scoring above this marks retrieval as **CORRECT** (internal docs only).
- `LOWER_TH = 0.3` — if all chunks score below this, retrieval is **INCORRECT** (web search only). Anything in between is **AMBIGUOUS** (docs + web).
- `search_kwargs={"k": 4}` — number of chunks retrieved from FAISS.
