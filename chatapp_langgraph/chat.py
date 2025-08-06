"""
LangGraph Chat Application with MongoDB Integration

This module provides a chat application with long-term memory, vector search,
and conversation persistence using MongoDB.
"""

import os
from datetime import datetime
from operator import add
from typing import Annotated, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables
load_dotenv()

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING", "mongodb://127.0.0.1:27017?directConnection=true"
)
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "langgraph_chat")
MONGODB_CHECKPOINTS_COLLECTION = os.getenv(
    "MONGODB_CHECKPOINTS_COLLECTION", "checkpoints_sync"
)
MONGODB_WRITES_COLLECTION = os.getenv("MONGODB_WRITES_COLLECTION", "writes_sync")
MONGODB_VECTOR_COLLECTION = os.getenv("MONGODB_VECTOR_COLLECTION", "vector_store_sync")
MONGODB_STORE_COLLECTION = os.getenv("MONGODB_STORE_COLLECTION", "long_term_memory_sync")


# State definition
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    user_id: str
    memory_summary: str


class LangGraphChatApp:
    def __init__(self):
        # Initialize MongoDB client
        self.mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
        self.db = self.mongo_client[MONGODB_DATABASE]

        # Database persists between sessions

        # Initialize components
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY
        )

        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        dimensions = len(self.embeddings.embed_query("foo"))

        # Initialize MongoDB Vector Search
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.db[MONGODB_VECTOR_COLLECTION],
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
            dimensions=dimensions,
        )

        # Initialize MongoDB Saver for checkpoints
        self.checkpointer = MongoDBSaver(
            client=self.mongo_client,
            db_name=MONGODB_DATABASE,
            collection_name=MONGODB_CHECKPOINTS_COLLECTION,
            writes_collection_name=MONGODB_WRITES_COLLECTION,
            auto_index_timeout=120,
        )

        # Initialize MongoDB Store for long-term memory (todo - play with index_config / vector index)
        self.memory_store = MongoDBStore(
            collection=self.db[MONGODB_STORE_COLLECTION],
        )

        # Build the graph
        self.app = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChatState)

        # Add nodes
        workflow.add_node("load_memory", self._load_memory)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)

        # Add edges
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)

        # Compile with checkpointer and store
        return workflow.compile(
            checkpointer=self.checkpointer,
            store=self.memory_store,  # this is injected in nodes that add store as kwarg
        )

    def _load_memory(self, state: ChatState, *, store: BaseStore) -> ChatState:
        """Load long-term memory for the user"""
        user_id = state.get("user_id", "default_user")

        # Load user's memory summary
        memory_data = store.get(namespace=("summaries",), key=user_id)
        memory_summary = ""

        if memory_data:
            memory_summary = memory_data.value["summary"]
            print(
                f"(Loaded memory for user {user_id}: {len(memory_summary)} characters)"
            )

        return {"memory_summary": memory_summary}

    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context using RAG"""
        # Get the latest human message
        latest_message = state["messages"][-1]
        if isinstance(latest_message, HumanMessage):
            query = latest_message.content

            # Perform similarity search
            docs = self.vector_store.similarity_search(
                query,
                k=3,  # Retrieve top 3 relevant documents
            )

            # Combine retrieved documents into context
            context = "\n\n".join([doc.page_content for doc in docs])

            return {
                "messages": [],
                "context": context,
                "user_id": state["user_id"],
                "memory_summary": state["memory_summary"],
            }
        else:
            raise RuntimeError(
                f"Received unexpected non-human message! {latest_message=}"
            )

    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response using LLM with retrieved context and memory"""
        messages = state["messages"]
        context = state.get("context", "")
        memory_summary = state.get("memory_summary", "")

        # Create system message with context and memory
        system_prompt = f"""You are a helpful AI assistant.

Previous conversation context about this user:
{memory_summary}

Current relevant information:
{context}

Use the previous context to maintain continuity in the conversation. If the current information is relevant to the user's question, incorporate it into your response. Answer naturally and conversationally."""

        # Prepare messages for LLM
        llm_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in messages:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})

        # Generate AI response
        ai_message = self.llm.invoke(llm_messages)

        return {"messages": [ai_message]}

    def _update_memory(self, state: ChatState, store: BaseStore) -> ChatState:
        """Update long-term memory with conversation insights"""
        user_id = state["user_id"]
        messages = state["messages"]
        current_memory = state.get("memory_summary", "")

        # Only update memory if we have meaningful conversation
        if len(messages) >= 2:  # At least one exchange
            # Get recent conversation context
            recent_messages = messages[-4:]  # Last 2 exchanges
            conversation_text = ""
            for msg in recent_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                conversation_text += f"{role}: {msg.content}\n"

            # Create prompt to update memory
            memory_prompt = f"""Current memory summary for this user:
{current_memory}

Recent conversation:
{conversation_text}

Please update the memory summary to include any important information about the user's interests, preferences, ongoing projects, or context that would be helpful for future conversations. Keep it concise but informative. If no significant new information, return the current summary unchanged."""

            # Generate updated memory
            memory_response = self.llm.invoke(
                [{"role": "user", "content": memory_prompt}]
            )

            updated_memory = memory_response.content

            # Get the number of conversations for a user
            conversation_count = 0
            if memory_data := store.get(namespace=("summaries",), key=user_id):
                conversation_count = memory_data.value["conversation_count"]

            # Store updated memory
            memory_data = {
                "summary": updated_memory,
                "last_updated": datetime.now().isoformat(),
                "conversation_count": conversation_count + 1,
            }

            store.put(
                namespace=("summaries",), key=user_id, value=memory_data, index=None
            )
            print(f"(Updated memory for user {user_id})")



    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadatas)
        ]

        self.vector_store.add_documents(docs)
        print(f"Added {len(docs)} documents to vector store")

    def chat(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: Optional[str] = None,
    ):
        """Send a message and get response"""
        if thread_id is None:
            thread_id = f"{user_id}_session"

        # Create human message
        human_message = HumanMessage(content=message)

        # Initial state
        initial_state = {
            "messages": [human_message],
            "context": "",
            "user_id": user_id,
            "memory_summary": "",
        }

        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(initial_state, config)

        # Return the AI response
        return result["messages"][-1].content

    def get_user_memory(self, user_id: str) -> dict:
        """Get the long-term memory for a user"""
        memory_data = self.memory_store.get(namespace=("summaries",), key=user_id)
        return memory_data or {}

    def clear_user_memory(self, user_id: str):
        """Clear the long-term memory for a user"""
        self.memory_store.delete(namespace=("summaries",), key=user_id)
        print(f"Cleared memory for user {user_id}")

    def list_user_sessions(self, user_id: str) -> list:
        """List all session thread IDs for a user"""
        # This is a simplified implementation
        # In practice, you might want to store session metadata in the store
        sessions = []
        prefix = f"{user_id}_session"

        # You could enhance this by storing session metadata
        # For now, we'll return the default session
        sessions.append(prefix)
        return sessions

    def get_chat_history(
        self, user_id: str = "default_user", thread_id: Optional[str] = None
    ):
        """Get chat history for a thread"""
        if thread_id is None:
            thread_id = f"{user_id}_session"

        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        return state.values.get("messages", []) if state.values else []

    def start_new_session(self, user_id: str) -> str:
        """Start a new chat session for a user"""
        import uuid

        new_thread_id = f"{user_id}_session_{uuid.uuid4().hex[:8]}"
        return new_thread_id

    def clear_database(self) -> None:
        """Clear all collections in the database (admin function)"""
        collections_cleared = []
        for coll_name in self.db.list_collection_names():
            self.db[coll_name].drop()
            collections_cleared.append(coll_name)
        print(f"Database '{MONGODB_DATABASE}' cleared. Collections removed: {collections_cleared}")

    def list_all_users(self) -> list[str]:
        """List all users who have memory stored"""
        try:
            return self.memory_store.collection.distinct("key", {"namespace": ["summaries"]})
        except Exception:
            return []