# 📚 Research Paper Answer Bot — RAG Capstone Project

## Author: Rajkumar KK
## Program: GenAI Pinnacle Plus Program (Analytics Vidhya)

---

## 📋 Project Overview

An intelligent chatbot that answers questions about Generative AI research papers using RAG (Retrieval-Augmented Generation). The system indexes seminal papers, retrieves relevant context, and generates grounded answers with source citations.

## 🏗️ Architecture

```
PDF Papers → Document Loading → Chunking → Embedding → Vector DB (ChromaDB)
                                                              ↓
User Query → Query Embedding → Retrieval (Hybrid) → Reranking → LLM (Gemini) → Answer + Citations
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain (LCEL) |
| Embeddings (Open-Source) | BAAI/bge-m3 |
| Embeddings (Commercial) | Google text-embedding-004 |
| Vector Store | ChromaDB |
| LLM | Google Gemini 2.0 Flash |
| Retrieval | Hybrid (BM25 + Dense) + Reranker |
| Evaluation | RAGAS |
| UI | Streamlit |

## 📁 Project Structure

```
capstone/
├── notebook_code.py          # Main notebook (Jupyter-compatible with # %% markers)
├── app.py                    # Streamlit application (Stretch Goal)
├── presentation.pptx         # 15-slide presentation
├── requirements.txt          # Python dependencies
├── .env                      # API key configuration
├── README.md                 # This file
└── data/papers/              # Research papers (downloaded by notebook)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your-actual-api-key-here
```

Get a free key at: https://aistudio.google.com/apikey

### 3. Run the Notebook

**Option A: VS Code (recommended)**
- Open `notebook_code.py` in VS Code
- Cells are separated by `# %%` markers
- Run cells interactively with Ctrl+Enter

**Option B: Convert to .ipynb**
```bash
pip install jupytext
jupytext --to notebook notebook_code.py
jupyter notebook notebook_code.ipynb
```

**Option C: JupyterLab**
- JupyterLab natively supports percent-format scripts
- Open `notebook_code.py` → Right-click → Open as Notebook

### 4. Run Streamlit App (Stretch Goal)

```bash
streamlit run app.py
```

## ✅ Submission Checklist

- [x] Notebook runs end-to-end without errors
- [x] ≥2 embedding models compared (BGE-M3 vs Google text-embedding-004)
- [x] Vector DB setup (ChromaDB) with all documents indexed
- [x] ≥2 retrieval strategies compared (Cosine, Hybrid, Reranker)
- [x] Complete RAG chain with LLM integration
- [x] Top-3 sources per answer (paper title + page number)
- [x] Stretch Goal: Streamlit UI + Conversational Memory
- [x] 15-slide presentation with architecture diagram
- [x] RAGAS evaluation metrics

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.89 |
| Answer Relevance | 0.85 |
| Context Precision | 0.82 |

## 📝 Research Papers Used

1. **Attention Is All You Need** (Vaswani et al., 2017)
2. **BERT** (Devlin et al., 2019)
3. **GPT-3: Language Models are Few-Shot Learners** (Brown et al., 2020)
4. **LLaMA: Open and Efficient Foundation Models** (Touvron et al., 2023)
5. **Retrieval-Augmented Generation** (Lewis et al., 2020)

## 📧 Contact

- **Name:** Rajkumar KK
- **Email:** kkrajkumarece@gmail.com
- **Program:** GenAI Pinnacle Plus (Analytics Vidhya)
