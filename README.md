# CLARA-AI

This project implements a **LangGraph + LangChain** agent framework for biomedical research assistance.  
It integrates **retrieval-augmented generation (RAG)**, **SQL querying**, and **planning/replanning workflows** to provide explainable, tool-using answers with citations.

---

## 🚀 Features

- **Graph-Oriented Agent Workflow**  
  Uses LangGraph to orchestrate planning, execution, and replanning steps for complex queries.

- **Retrieval-Augmented Generation (RAG)**  
  Connects to a PostgreSQL + PGVector database of biomedical literature with self-query and contextual compression.

- **SQL Database Access (Optional)**  
  Supports structured queries against relational databases using LangChain’s SQL toolkit.

- **Adaptive Planner & Replanner**  
  Plans multi-step tool usage, replans when necessary, and provides structured final responses with inline citations.

- **Configurable with Environment Variables**  
  Load secrets and configuration via `.env`.

- **Chainlit Integration**  
  Provides a chat interface with authentication, session management, and streaming answers.

---

## 📂 Repository Structure

```
core/
  config.py        # Environment configuration
  graph.py         # Agent graph definition
  helper.py        # Utility functions & callback handlers
  prompts.py       # System, planner, and replanner prompts
  retriever.py     # PGVector retriever tool with compression
app.py             # Chainlit entry point
requirements.txt   # Python dependencies
```

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key
SQL_DATABASE_URL=postgresql+psycopg://user:password@host:port/dbname
PGVECTOR_URL=postgresql+psycopg://user:password@host:port/dbname
PGVECTOR_COLLECTION=biomed_vectors
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
TOP_K=10
```

### 5. Run the App
```bash
chainlit run app.py -w
```
Then open the provided local URL in your browser.

---

## 🧩 How It Works

1. **Planner**  
   Generates a minimal executable plan of tool calls and reasoning steps.

2. **Executor (Agent)**  
   Executes each step, calling RAG or SQL tools as required.

3. **Replanner / Decider**  
   Revises the plan if gaps remain, or finalizes a response with citations.

4. **Retriever Tool**  
   Queries PGVector with self-querying + contextual compression to return the most relevant passages and metadata.

5. **Chainlit UI**  
   Provides an interactive chat interface, showing final answers and their sources.

---

## 🔧 Customization

- **Prompts**: Modify `core/prompts.py` to change planning, retrieval, and execution behavior.
- **Retriever**: Adjust metadata schema and compression pipeline in `core/retriever.py`.
- **Config**: Update `.env` or `AppConfig` in `core/config.py` for new backends, models, or parameters.

---

## 📜 Requirements

See [`requirements.txt`](requirements.txt):
- `langchain`, `langgraph`, `langchain-openai`, `langchain-community`
- `psycopg[binary,pool]`, `pgvector`, `SQLAlchemy`
- `pydantic`, `python-dotenv`
- `chainlit`, `streamlit`

---

## 🛡️ Notes

- Ensure your PostgreSQL instance has the **pgvector extension** installed.  
- Sensitive information (API keys, database credentials) should only be stored in `.env`.  
- SQL database integration is optional — if no `SQL_DATABASE_URL` is provided, only RAG is used.  

---

## 📖 Example Use Case

**Query**: *“What are recent PICOS-compliant trials on brain-heart interconnectome?”*  
**Workflow**:
1. Planner generates steps to retrieve from PGVector and filter by compliance.
2. Executor calls the retriever tool.
3. Replanner synthesizes the final answer with inline citations.
4. Chainlit displays the response and a clickable list of sources.

---

## 📜 License
This project is licensed under the **MIT License**. See `LICENSE` for details.
