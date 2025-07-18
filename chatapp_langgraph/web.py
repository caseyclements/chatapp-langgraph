"""
Web interface for Atlas Chat application.

This module provides a FastAPI-based web service for interacting with
the LangGraph chat application via REST API endpoints.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys

from .chat import LangGraphChatApp


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str


class AddDocumentsRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class AddDocumentsResponse(BaseModel):
    message: str
    count: int


class MemoryResponse(BaseModel):
    user_id: str
    memory: Optional[Dict[str, Any]] = None
    has_memory: bool


class UsersResponse(BaseModel):
    users: List[str]
    count: int


class DatabaseResponse(BaseModel):
    message: str
    collections_cleared: Optional[List[str]] = None


class StatusResponse(BaseModel):
    status: str
    message: str


# Global app instance
chat_app = None


def get_chat_app() -> LangGraphChatApp:
    """Get or create the chat application instance"""
    global chat_app
    if chat_app is None:
        chat_app = LangGraphChatApp()
    return chat_app


# FastAPI app
app = FastAPI(
    title="Atlas Chat API",
    description="REST API for Atlas Chat - LangGraph + MongoDB chat application",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Atlas Chat</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .chat-area {
                height: 400px;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                overflow-y: auto;
                background-color: #fafafa;
                margin-bottom: 20px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
            }
            .user-message {
                background-color: #3498db;
                color: white;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #ecf0f1;
                margin-right: 20%;
            }
            .input-area {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            input[type="text"] {
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
            }
            button {
                padding: 12px 24px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .actions {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 20px;
            }
            .action-btn {
                padding: 8px 16px;
                background-color: #95a5a6;
                font-size: 14px;
            }
            .action-btn:hover {
                background-color: #7f8c8d;
            }
            .user-input {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .user-input input {
                flex: 1;
            }
            .docs-area {
                display: none;
                margin-top: 20px;
            }
            .docs-area textarea {
                width: 100%;
                height: 150px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-family: monospace;
                resize: vertical;
            }
            .api-links {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
            }
            .api-links a {
                color: #3498db;
                text-decoration: none;
                margin: 0 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Atlas Chat</h1>
            
            <div class="user-input">
                <input type="text" id="userId" placeholder="Enter your name" value="WebUser">
                <button onclick="setUser()">Set User</button>
            </div>
            
            <div class="chat-area" id="chatArea">
                <div class="message bot-message">
                    <strong>🤖 Atlas Chat:</strong> Welcome! Enter your name above and start chatting.
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
            
            <div class="actions">
                <button class="action-btn" onclick="showMemory()">Show Memory</button>
                <button class="action-btn" onclick="showUsers()">List Users</button>
                <button class="action-btn" onclick="toggleDocs()">Add Documents</button>
                <button class="action-btn" onclick="clearDatabase()" style="background-color: #e74c3c;">Clear Database</button>
            </div>
            
            <div class="docs-area" id="docsArea">
                <h3>Add Documents</h3>
                <textarea id="docsInput" placeholder="Enter documents, one per line..."></textarea>
                <button onclick="addDocuments()">Add Documents</button>
            </div>
            
            <div class="api-links">
                <a href="/docs" target="_blank">API Documentation</a> |
                <a href="/redoc" target="_blank">ReDoc</a>
            </div>
        </div>

        <script>
            let currentUser = 'WebUser';
            
            function setUser() {
                const userId = document.getElementById('userId').value.trim();
                if (userId) {
                    currentUser = userId;
                    addMessage('System', `User set to: ${currentUser}`, 'bot-message');
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function addMessage(sender, message, className) {
                const chatArea = document.getElementById('chatArea');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${className}`;
                messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
            
            async function sendMessage() {
                const messageInput = document.getElementById('messageInput');
                const message = messageInput.value.trim();
                
                if (!message) return;
                
                addMessage(`👤 ${currentUser}`, message, 'user-message');
                messageInput.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            user_id: currentUser
                        })
                    });
                    
                    const data = await response.json();
                    addMessage('🤖 Assistant', data.response, 'bot-message');
                } catch (error) {
                    addMessage('❌ Error', 'Failed to send message: ' + error.message, 'bot-message');
                }
            }
            
            async function showMemory() {
                try {
                    const response = await fetch(`/memory/${currentUser}`);
                    const data = await response.json();
                    
                    if (data.has_memory && data.memory) {
                        const summary = data.memory.summary || 'No summary available';
                        const updated = data.memory.last_updated || 'Unknown';
                        const count = data.memory.conversation_count || 0;
                        
                        addMessage('🧠 Memory', `Summary: ${summary}\\nLast updated: ${updated}\\nConversations: ${count}`, 'bot-message');
                    } else {
                        addMessage('🧠 Memory', 'No memory found for this user', 'bot-message');
                    }
                } catch (error) {
                    addMessage('❌ Error', 'Failed to get memory: ' + error.message, 'bot-message');
                }
            }
            
            async function showUsers() {
                try {
                    const response = await fetch('/users');
                    const data = await response.json();
                    
                    if (data.users.length > 0) {
                        addMessage('👥 Users', `Found ${data.count} users: ${data.users.join(', ')}`, 'bot-message');
                    } else {
                        addMessage('👥 Users', 'No users found with stored conversations', 'bot-message');
                    }
                } catch (error) {
                    addMessage('❌ Error', 'Failed to get users: ' + error.message, 'bot-message');
                }
            }
            
            function toggleDocs() {
                const docsArea = document.getElementById('docsArea');
                docsArea.style.display = docsArea.style.display === 'none' ? 'block' : 'none';
            }
            
            async function addDocuments() {
                const docsInput = document.getElementById('docsInput');
                const docsText = docsInput.value.trim();
                
                if (!docsText) return;
                
                const documents = docsText.split('\\n').filter(doc => doc.trim().length > 0);
                
                try {
                    const response = await fetch('/documents', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            documents: documents
                        })
                    });
                    
                    const data = await response.json();
                    addMessage('📄 Documents', data.message, 'bot-message');
                    docsInput.value = '';
                    toggleDocs();
                } catch (error) {
                    addMessage('❌ Error', 'Failed to add documents: ' + error.message, 'bot-message');
                }
            }
            
            async function clearDatabase() {
                if (!confirm('⚠️ WARNING: This will clear ALL data in the database! Are you sure?')) {
                    return;
                }
                
                try {
                    const response = await fetch('/database', {
                        method: 'DELETE'
                    });
                    
                    const data = await response.json();
                    addMessage('🗑️ Database', data.message, 'bot-message');
                } catch (error) {
                    addMessage('❌ Error', 'Failed to clear database: ' + error.message, 'bot-message');
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return StatusResponse(status="healthy", message="Atlas Chat API is running")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Send a chat message and get response"""
    try:
        chat_app = get_chat_app()
        
        # Use provided session_id or create new one
        session_id = request.session_id
        if not session_id:
            session_id = chat_app.start_new_session(request.user_id)
        
        response = chat_app.chat(request.message, request.user_id, session_id)
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    """Get user's conversation memory"""
    try:
        chat_app = get_chat_app()
        memory = chat_app.get_user_memory(user_id)
        
        if memory and hasattr(memory, 'value'):
            return MemoryResponse(
                user_id=user_id,
                memory=memory.value,
                has_memory=True
            )
        else:
            return MemoryResponse(
                user_id=user_id,
                memory=None,
                has_memory=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory error: {str(e)}")


@app.post("/documents", response_model=AddDocumentsResponse)
async def add_documents(request: AddDocumentsRequest):
    """Add documents to the knowledge base"""
    try:
        chat_app = get_chat_app()
        chat_app.add_documents(request.documents, request.metadatas)
        
        return AddDocumentsResponse(
            message=f"Successfully added {len(request.documents)} documents to the knowledge base",
            count=len(request.documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document error: {str(e)}")


@app.get("/users", response_model=UsersResponse)
async def list_users():
    """List all users with stored conversations"""
    try:
        chat_app = get_chat_app()
        users = chat_app.list_all_users()
        
        return UsersResponse(
            users=users,
            count=len(users)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Users error: {str(e)}")


@app.delete("/database", response_model=DatabaseResponse)
async def clear_database():
    """Clear all data in the database (admin function)"""
    try:
        chat_app = get_chat_app()
        
        # Get collections before clearing
        collections = list(chat_app.db.list_collection_names())
        
        # Clear database
        chat_app.clear_database()
        
        return DatabaseResponse(
            message=f"Database cleared successfully. Removed {len(collections)} collections.",
            collections_cleared=collections
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


def main():
    """Entry point for the web service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Atlas Chat Web Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🌍 Starting Atlas Chat Web Service...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "chatapp_langgraph.web:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Atlas Chat Web Service...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()