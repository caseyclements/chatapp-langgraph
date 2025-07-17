# LangGraph Chat App with MongoDB

This project contains two demo chat applications built with LangGraph and MongoDB integration:

- `chatapp_with_memory.py` - Synchronous version with long-term memory
- `chatapp_async_in_executor.py` - Asynchronous version with enhanced features

## Features

- **Long-term Memory**: Persists user conversation context across sessions
- **Vector Search**: RAG (Retrieval-Augmented Generation) with MongoDB Atlas Vector Search
- **Conversation Persistence**: Chat history stored in MongoDB
- **Multi-user Support**: Separate memory and sessions per user

## Setup

1. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and MongoDB connection string
   ```

3. **MongoDB Setup**:
   - Ensure MongoDB is running locally or configure Atlas connection
   - The apps will create necessary collections automatically

## Usage

### Run the synchronous version:
```bash
uv run python chatapp_with_memory.py
```

### Run the asynchronous version:
```bash
uv run python chatapp_async_in_executor.py
```

## Dependencies

- **LangChain**: Core framework for LLM applications
- **LangGraph**: State management and workflow orchestration
- **MongoDB**: Document database with vector search capabilities
- **OpenAI**: LLM provider for chat completions and embeddings

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGODB_CONNECTION_STRING`: MongoDB connection string
- `MONGODB_DATABASE`: Database name (default: langgraph_chat)
- Collection names can be customized via additional environment variables