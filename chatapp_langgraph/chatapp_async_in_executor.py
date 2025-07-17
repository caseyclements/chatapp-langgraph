"""This ASync version attempts to use our Stores with Synchronous Client stack."""

import asyncio
import os
from datetime import datetime
from operator import add
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

from langchain_mongodb import MongoDBAtlasVectorSearch

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING", "mongodb://127.0.0.1:27017?directConnection=true"
)
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "langgraph_chat")
MONGODB_CHECKPOINTS_COLLECTION = os.getenv(
    "MONGODB_CHECKPOINTS_COLLECTION", "checkpoints_executor"
)
MONGODB_WRITES_COLLECTION = os.getenv("MONGODB_WRITES_COLLECTION", "writes_executor")
MONGODB_VECTOR_COLLECTION = os.getenv(
    "MONGODB_VECTOR_COLLECTION", "vector_store_executor"
)
MONGODB_STORE_COLLECTION = os.getenv(
    "MONGODB_STORE_COLLECTION", "long_term_memory_executor"
)


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

        # Initialize components
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API_KEY, seed=0,
        )

        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        dimensions = len(self.embeddings.embed_query("foo"))

        # Initialize MongoDB Vector Search
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            MONGODB_CONNECTION_STRING,
            namespace=f"{MONGODB_DATABASE}.{MONGODB_VECTOR_COLLECTION}",
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
            dimensions=dimensions,
        )
        # self.vector_store = MongoDBAtlasVectorSearch(
        # collection=self.db[MONGODB_VECTOR_COLLECTION],
        # embedding=self.embeddings,
        # index_name="vector_index",
        # text_key="text",
        # embedding_key="embedding"
        # )

        # Initialize MongoDB Saver for checkpoints
        self.checkpointer = MongoDBSaver(
            client=self.mongo_client,
            db_name=MONGODB_DATABASE,
            checkpoint_collection_name=MONGODB_CHECKPOINTS_COLLECTION,
            writes_collection_name=MONGODB_WRITES_COLLECTION,
            auto_index_timeout=120,
        )

        # Initialize MongoDB Store for long-term memory
        self.memory_store = MongoDBStore(
            collection=self.db[MONGODB_STORE_COLLECTION],
            auto_index_timeout=120,
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
        return workflow.compile(checkpointer=self.checkpointer, store=self.memory_store)

    async def _load_memory(self, state: ChatState) -> ChatState:
        """Load long-term memory for the user"""
        user_id = state.get("user_id", "default_user")

        # Load user's memory summary
        memory_data = await self.memory_store.aget(namespace=("summaries",), key=user_id)
        memory_summary = ""

        if memory_data:
            memory_summary = memory_data.value.get("summary", "")
            print(f"Loaded memory for user {user_id}: {len(memory_summary)} characters")

        return {"memory_summary": memory_summary}

    async def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context using RAG"""
        # Get the latest human message
        print(f"\n\nIn retrieve context: messages ({len(state['messages'])}:{state['messages']}")
        latest_message = state["messages"][-1]
        if isinstance(latest_message, HumanMessage):
            query = latest_message.content

            # Perform similarity search
            docs = await self.vector_store.asimilarity_search(
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

    async def _generate_response(self, state: ChatState) -> ChatState:
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

        # Create AI message
        ai_message = await self.llm.ainvoke(llm_messages)

        return {"messages": [ai_message]}


    async def _update_memory(self, state: ChatState) -> ChatState:
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
            memory_response = await self.llm.ainvoke(
                [{"role": "user", "content": memory_prompt}]
            )

            updated_memory = memory_response.content

            # Store updated memory
            memory_data = {
                "summary": updated_memory,
                "last_updated": datetime.now().isoformat(),
                "conversation_count": await self._get_conversation_count(user_id) + 1,
            }

            await self.memory_store.aput(
                namespace=("summaries",), key=user_id, value=memory_data, index=None
            )
            print(f"Updated memory for user {user_id}")

    async def _get_conversation_count(self, user_id: str) -> int:
        """Get the number of conversations for a user"""
        memory_data = await self.memory_store.aget(namespace=("summaries",), key=user_id)
        if memory_data:
            return memory_data.value.get("conversation_count", 0)  # TODO - Confirm this is as expected.
        return 0

    async def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadatas)
        ]

        await self.vector_store.aadd_documents(docs)
        print(f"Added {len(docs)} documents to vector store")

    async def chat(
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
        result = await self.app.ainvoke(initial_state, config)

        # Return the AI response
        return result["messages"][-1].content

    async def get_user_memory(self, user_id: str) -> dict:
        """Get the long-term memory for a user"""
        memory_data = await self.memory_store.aget(
            namespace=("summaries",), key=user_id
        )
        return memory_data or {}

    async def clear_user_memory(self, user_id: str):
        """Clear the long-term memory for a user"""
        await self.memory_store.adelete(namespace=("summaries",), key=user_id)
        print(f"Cleared memory for user {user_id}")

    async def list_user_sessions(self, user_id: str) -> list:
        """List all session thread IDs for a user"""
        # This is a simplified implementation
        # In practice, you might want to store session metadata in the store
        sessions = []
        prefix = f"{user_id}_session"

        # You could enhance this by storing session metadata
        # For now, we'll return the default session
        sessions.append(prefix)
        return sessions

    async def get_chat_history(
        self, user_id: str = "default_user", thread_id: Optional[str] = None
    ):
        """Get chat history for a thread"""
        if thread_id is None:
            thread_id = f"{user_id}_session"

        config = {"configurable": {"thread_id": thread_id}}
        state = await self.app.aget_state(config)
        return state.values.get("messages", []) if state.values else []

    async def start_new_session(self, user_id: str) -> str:
        """Start a new chat session for a user"""
        import uuid

        new_thread_id = f"{user_id}_session_{uuid.uuid4().hex[:8]}"
        return new_thread_id



async def main():
    """Example usage of ChatApp in asynchronous mode."""

        # Create two users
    user_id1 = "john_doe"
    user_id2 = "jane_smith"

    # Initialize the chat app
    chat_app = LangGraphChatApp()

    await chat_app.clear_user_memory(user_id=user_id1)  # todo - unnecessary
    await chat_app.clear_user_memory(user_id=user_id2)

    # todo test code. I'd prefer not to have to manually clear the checkpointer
    chat_app.checkpointer.checkpoint_collection.delete_many({})
    chat_app.checkpointer.writes_collection.delete_many({})

    # Example: Add some documents to the vector store
    sample_docs = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "MongoDB Atlas Vector Search enables you to store and search vector embeddings.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
        "Long-term memory allows AI assistants to remember user preferences and context across sessions.",
    ]

    await chat_app.add_documents(sample_docs)

    # Example conversation with user persistence
    print("\n=== First Conversation ===")
    query1 = "Hi, I'm working on a Python project with LangGraph"
    print(f"{user_id1}: {query1}")
    response1 = await chat_app.chat(query1, user_id1)
    print(f"Assistant: {response1}")

    query2 = "I'm particularly interested in using MongoDB for persistence"
    print(f"{user_id1}: {query2}")
    response2 = await chat_app.chat(query2, user_id1)
    print(f"Assistant: {response2}")

    # Check memory after first conversation
    memory = await chat_app.get_user_memory(user_id1)
    print(f"\nUser memory after first conversation: {memory}")

    print("\n=== Second Conversation (Later Session) ===")
    # Start a new session but same user - memory should persist
    new_session = await chat_app.start_new_session(user_id1)

    query3 = "What was I working on before?"
    print(f"{user_id1}: {query3}")
    response3 = await chat_app.chat(query3, user_id1, new_session)
    print(f"Assistant: {response3}")

    query4 = "Can you help me with the MongoDB integration?"
    print(f"{user_id1}: {query4}")
    response4 = await chat_app.chat(query4, user_id1, new_session)
    print(f"Assistant: {response4}")

    # Check updated memory
    updated_memory = await chat_app.get_user_memory(user_id1)
    print(f"\nUpdated user memory: {updated_memory}")

    # Show different user
    print("\n=== Different User ===")
    query5 = "Hello, I'm new here. Do you know anything about me?"
    print(f"{user_id2}: {query5}")
    response5 = await chat_app.chat(query5, user_id2)
    print(f"Assistant: {response5}")

    print("\n=== State of the Memory Store at the end ===")
    print(f"{chat_app.memory_store.list_namespaces() = }")
    print(f"{chat_app.memory_store.collection.distinct(key="key", filter={"namespace": ["summaries"]}) = }")
    print(f"summaries: {chat_app.memory_store.collection.find({"namespace": ["summaries"]}, {"key": 1, "value.summary": 1, "_id": 0}).to_list()}")



if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    print("\n=== End demo ===")
