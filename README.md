# LangGraph Chat App with MongoDB

This project provides Atlas Chat - an AI assistant application built with LangGraph and MongoDB integration, available as both a terminal and web interface.

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

### Run the web service:
```bash
uv run atlas-chat-web
```
- Web interface: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Alternative docs: http://127.0.0.1:8000/redoc

### Run the terminal chat application:
```bash
uv run atlas-chat
```

### Run the example script:
```bash
uv run examples/chat_sync.py
```

### Available Commands in Atlas Chat (Terminal):
- `/help` - Show available commands
- `/memory` - View your conversation memory
- `/add_docs` - Add documents to the knowledge base
- `/users` - List all users (admin)
- `/clear_db` - Clear entire database (admin)
- `/quit` or `/exit` - Exit the application
- Just type normally to chat!

### Web Service API Endpoints:
- `POST /chat` - Send chat messages
- `GET /memory/{user_id}` - Get user's conversation memory
- `POST /documents` - Add documents to knowledge base
- `GET /users` - List all users
- `DELETE /database` - Clear database (admin)
- `GET /health` - Health check

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