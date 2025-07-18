"""
Synchronous Chat Application Example

This example demonstrates how to use the LangGraphChatApp for building
a chat application with long-term memory and vector search capabilities.

See also: https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent/

Scope for expanding:
- long-term memory
    - how do we incorporate different memories of a single memory?
        - topics => this can be embedded for chatbot to get relevant memories
        - summary => summarize conversation for a single user
"""

from chatapp_langgraph.chat import LangGraphChatApp


def main():
    """Example usage and setup"""
    
    # Create two users
    user_id1 = "john_doe"
    user_id2 = "jane_smith"

    # Initialize the chat app
    chat_app = LangGraphChatApp()

    # Example: Add some documents to the vector store
    sample_docs = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "MongoDB Atlas Vector Search enables you to store and search vector embeddings.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
        "Long-term memory allows AI assistants to remember user preferences and context across sessions.",
    ]

    chat_app.add_documents(sample_docs)

    # Example conversation with user persistence
    print("=== First Conversation ===")
    query1 = "Hi, I'm working on a Python project with LangGraph"
    print(f"{user_id1}: {query1}")
    response1 = chat_app.chat(query1, user_id1)
    print(f"Assistant: {response1}")
    
    query2 = "I'm particularly interested in using MongoDB for persistence"
    print(f"{user_id1}: {query2}")
    response2 = chat_app.chat(query2, user_id1)
    print(f"Assistant: {response2}")

    # Check memory after first conversation
    memory = chat_app.get_user_memory(user_id1)
    print(f"\n==>User memory after first conversation: {memory}")

    print("\n=== Second Conversation (Later Session) ===")
    # Start a new session but same user - memory should persist
    new_session = chat_app.start_new_session(user_id1)

    query3 = "What was I working on before?"
    print(f"{user_id1}: {query3}")
    response3 = chat_app.chat(query3, user_id1, new_session)
    print(f"Assistant: {response3}")

    query4 = "Can you help me with the MongoDB integration?"
    print(f"{user_id1}: {query4}")
    response4 = chat_app.chat(query4, user_id1, new_session)
    print(f"Assistant: {response4}")

    # Check updated memory
    updated_memory = chat_app.get_user_memory(user_id1)
    print(f"\n(Updated user memory: {updated_memory})")

    # Show different user
    print("\n=== Different User ===")
    query5 = "Hello, I'm new here. Do you know anything about me?"
    print(f"{user_id2}: {query5}")
    response5 = chat_app.chat(query5, user_id2)
    print(f"Assistant: {response5}")

    print("\n=== State of the Memory Store at the end ===")
    print(f"{chat_app.memory_store.list_namespaces() = }")
    print(f"{chat_app.memory_store.collection.distinct(key='key', filter={'namespace': ['summaries']}) = }")
    print(f"summaries: {chat_app.memory_store.collection.find({'namespace': ['summaries']}, {'key': 1, 'value.summary': 1, '_id': 0}).to_list()}")


if __name__ == "__main__":
    # Run the example
    main()
    print("End demo.")

# TODO
#   - Adjust this to take user input
#   - test by running from the shell
#   - related: serve as a web site