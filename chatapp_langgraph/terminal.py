"""
Terminal interface for Atlas Chat application.

This module provides a simple terminal-based interface for interacting with
the LangGraph chat application using a command-line interface.
"""

import os
import sys
from typing import Optional

from .chat import LangGraphChatApp


class AtlasChatTerminal:
    """Terminal interface for Atlas Chat"""
    
    def __init__(self):
        self.app = LangGraphChatApp()
        self.current_user = None
        self.current_session = None
    
    def display_welcome(self):
        """Display welcome message"""
        print("=" * 50)
        print("🌍 Welcome to Atlas Chat!")
        print("Your AI assistant powered by LangGraph & MongoDB")
        print("=" * 50)
        print()
    
    def get_user_name(self) -> str:
        """Get and validate user name"""
        while True:
            name = input("What's your name? ").strip()
            if name:
                return name
            print("Please enter a valid name.")
    
    def check_existing_user(self, name: str) -> bool:
        """Check if user exists in memory store"""
        all_users = self.app.list_all_users()
        return name in all_users
    
    def handle_new_user_flow(self, name: str) -> str:
        """Handle the new user authentication flow"""
        if self.check_existing_user(name):
            print(f"Hi {name}! Welcome back.")
            return name
        
        print(f"Hi {name}! I don't have any previous conversations with you.")
        
        while True:
            is_new = input("Are you a new user? (y/n): ").strip().lower()
            if is_new in ['y', 'yes']:
                print(f"Welcome to Atlas Chat, {name}!")
                return name
            elif is_new in ['n', 'no']:
                print("You mentioned you've spoken to us before. You might have used a different username.")
                
                while True:
                    try_different = input("Would you like to try a different name? (y/n): ").strip().lower()
                    if try_different in ['y', 'yes']:
                        existing_users = self.app.list_all_users()
                        if existing_users:
                            print(f"Here are users I know: {', '.join(existing_users)}")
                        old_name = input("What name did you use before? ").strip()
                        if old_name:
                            if self.check_existing_user(old_name):
                                print(f"Found you! Welcome back {old_name}.")
                                return old_name
                            else:
                                print(f"I don't find any previous conversations with '{old_name}'.")
                                continue
                    elif try_different in ['n', 'no']:
                        print(f"Okay, I'll treat you as a new user named {name}.")
                        return name
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def parse_command(self, user_input: str) -> tuple[str, str]:
        """Parse user input to determine command type and content"""
        user_input = user_input.strip()
        
        # Check for special commands
        if user_input.startswith('/'):
            parts = user_input[1:].split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return command, args
        
        # Regular chat message
        return "chat", user_input
    
    def handle_command(self, command: str, args: str) -> bool:
        """Handle different commands. Returns True to continue, False to exit"""
        if command == "help":
            self.display_help()
        elif command == "memory":
            self.show_user_memory()
        elif command == "add_docs":
            self.add_documents_interactive()
        elif command == "clear_db":
            self.clear_database()
        elif command == "users":
            self.list_users()
        elif command == "quit" or command == "exit":
            return False
        elif command == "chat":
            if args:
                self.handle_chat_message(args)
            else:
                print("Please provide a message to chat about.")
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.")
        
        return True
    
    def display_help(self):
        """Display help information"""
        print("\n🔧 Available Commands:")
        print("  /help          - Show this help message")
        print("  /memory        - Show your conversation memory")
        print("  /add_docs      - Add documents to the knowledge base")
        print("  /users         - List all users (admin)")
        print("  /clear_db      - Clear entire database (admin)")
        print("  /quit or /exit - Exit the application")
        print("  Just type normally to chat!")
        print()
    
    def show_user_memory(self):
        """Display user's memory summary"""
        memory = self.app.get_user_memory(self.current_user)
        if memory:
            print(f"\n🧠 Your Memory Summary:")
            if hasattr(memory, 'value') and 'summary' in memory.value:
                print(f"  {memory.value['summary']}")
                print(f"  Last updated: {memory.value.get('last_updated', 'Unknown')}")
                print(f"  Conversations: {memory.value.get('conversation_count', 0)}")
            else:
                print("  No detailed memory available")
        else:
            print("\n🧠 No memory found for your user account.")
        print()
    
    def add_documents_interactive(self):
        """Interactive document addition"""
        print("\n📄 Add Documents to Knowledge Base")
        print("Enter documents one per line. Empty line to finish:")
        
        documents = []
        while True:
            doc = input("> ").strip()
            if not doc:
                break
            documents.append(doc)
        
        if documents:
            self.app.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to the knowledge base.")
        else:
            print("No documents added.")
        print()
    
    def clear_database(self):
        """Clear the entire database (admin function)"""
        print("\n⚠️  WARNING: This will clear ALL data in the database!")
        confirm = input("Are you sure? Type 'YES' to confirm: ").strip()
        if confirm == "YES":
            self.app.clear_database()
            print("✅ Database cleared successfully.")
        else:
            print("❌ Database clear cancelled.")
        print()
    
    def list_users(self):
        """List all users with stored memory"""
        users = self.app.list_all_users()
        if users:
            print(f"\n👥 Users with stored conversations: {', '.join(users)}")
        else:
            print("\n👥 No users found with stored conversations.")
        print()
    
    def handle_chat_message(self, message: str):
        """Handle a chat message"""
        print(f"\n💬 {self.current_user}: {message}")
        
        try:
            response = self.app.chat(message, self.current_user, self.current_session)
            print(f"🤖 Assistant: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()
    
    def run(self):
        """Main application loop"""
        try:
            self.display_welcome()
            
            # User authentication
            name = self.get_user_name()
            self.current_user = self.handle_new_user_flow(name)
            self.current_session = self.app.start_new_session(self.current_user)
            
            print(f"\n🚀 Ready to chat! Type /help for commands or just start chatting.")
            print("Type /quit to exit.\n")
            
            # Main chat loop
            while True:
                try:
                    user_input = input(f"{self.current_user}> ").strip()
                    
                    if not user_input:
                        continue
                    
                    command, args = self.parse_command(user_input)
                    
                    if not self.handle_command(command, args):
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Application error: {e}")
            sys.exit(1)


def main():
    """Entry point for the atlas-chat terminal application"""
    terminal = AtlasChatTerminal()
    terminal.run()


if __name__ == "__main__":
    main()