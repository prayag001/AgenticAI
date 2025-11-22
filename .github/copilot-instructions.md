# AgenticAI Project - AI Agent Instructions

## Project Overview
AgenticAI is a learning workspace for building LLM-powered applications with multiple frameworks: **LangChain**, **LangGraph**, and **Autogen**. All code is in Jupyter notebooks organized by framework/topic.

## Environment & Authentication

### API Key Pattern (Critical)
**Always** load environment variables using this exact pattern:
```python
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("GITHUB_TOKEN")  # or other API key names
```

### Supported API Keys
- `GITHUB_TOKEN` - GitHub Models (primary, uses `https://models.github.ai/inference`)
- `OPENAI_API_KEY` - OpenAI direct
- `GROQ_API_KEY` - Groq models
- `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` - LangSmith tracing
- `GOOGLE_API_KEY` - Google Gemini

### GitHub Models Configuration
When using GitHub-hosted models (default preference):
```python
token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "gpt-4.1-mini"  # or "openai/gpt-4.1-mini" for LangChain

# For LangChain:
llm = ChatOpenAI(model=model, api_key=token, base_url=endpoint)

# For Autogen:
from autogen_ext.models.openai import OpenAIChatCompletionClient
model_client = OpenAIChatCompletionClient(model=model, api_key=token, base_url=endpoint)
```

## Framework-Specific Conventions

### Autogen (7-autogen/)
- **Async by default**: Always use `await` with agent operations
- Agent initialization: `AssistantAgent(name="...", model_client=model_client)`
- Run pattern: `result = await agent.run(task="...")`
- Use `OpenAIChatCompletionClient` from `autogen_ext.models.openai`

### LangChain (2-Langchain_Basics/, 3-LangChain/)
- Import models from specific packages: `langchain_openai`, `langchain_groq`, `langchain_google_genai`
- Use `.invoke()` method for single calls (not async in notebooks)
- Enable tracing: Set `os.environ["LANGCHAIN_TRACING_V2"]="true"`
- Chaining: Use pipe operator `|` (e.g., `prompt | llm | output_parser`)

### LangGraph (4-LangGraph/)
- Tools: `WikipediaQueryRun`, `YouTubeSearchTool`, `TavilySearchResults` (from `langchain_community.tools`)
- Tool initialization requires API wrappers (e.g., `WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500)`)
- Uses same model configuration as LangChain

## Vector Database & RAG Patterns

### Document Loading + Splitting
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("file.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
final_docs = text_splitter.split_documents(docs)
```

### Embeddings
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
query_result = embeddings.embed_query(text)
```

### Vector Stores in Use
- **FAISS** (`2-Langchain_Basics/Class_3/vectorStore/FAISS/`) - Local, persisted indexes
- **Pinecone** - Cloud-hosted (examples in `Class_4/vector_search_rag_app.ipynb`)
- **Chroma** - Alternative local option

## Notebook Workflow
1. **Class folders** are sequential learning modules (Class_1, Class_2, etc.)
2. **Assignment solutions** use numeric suffixes (`Assignment_Sol_1.ipynb`, `Assignment_Sol_2.ipynb`)
3. Data files (`.txt`, `.xml`) live in same directory as consuming notebooks
4. Run cells sequentially - notebooks maintain state dependencies

## Common Pitfalls
- **Don't forget `load_dotenv()`** - API calls will fail silently
- **Async in Autogen**: Missing `await` causes runtime errors
- **GitHub Models**: Use `gpt-4.1-mini` (not `gpt-4o-mini`)
- **LangChain base_url**: Set explicitly for non-OpenAI endpoints
- **Tool wrappers**: Always configure before creating tools (e.g., `WikipediaAPIWrapper` before `WikipediaQueryRun`)
