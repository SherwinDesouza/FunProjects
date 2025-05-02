import os
import smtplib
import pywhatkit
from typing import TypedDict, Annotated, Sequence, List
from typing import Dict, List, Any
import re
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import spacy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.embeddings import GPT4AllEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, MessageGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from datetime import datetime
from typing import Literal
from PyPDF2 import PdfReader


# Load environment variables
load_dotenv()
#CHROMA_PATH = r"C:\Users\PAX\LangGraphFolder\restaurant-chatbot\chroma_db"
embedding_model = GPT4AllEmbeddings()
PDF_PATH = r"C:\Users\PAX\LangGraphFolder\restaurant-chatbot\FastFoodMenu.pdf"
groq_api_key = os.getenv("groq_api_key")
nlp = spacy.load('en_core_web_sm')

# Initialize the model
model = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_api_key)

def detect_intent(message, chat_history):
    """
    Detect user intent by considering current message + conversation history.
    """
    # Merge the last few chat turns
    history_text = ""
    for turn in chat_history[-3:]:  # Last 3 turns
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    
    prompt = f"""
You are an intent detection bot.
Here is the conversation so far:
{history_text}
Now the user said: "{message}"

Based on the FULL conversation, what is the user's intent?
Possible intents:
- ask_menu (User is asking about menu items)
- return_deals (User is asking about deals, offers, combos)
- place_order (User wants to order something)
- cancel_order (User wants to cancel or says they don't want anything)
- goodbye (User says bye, thanks, or signals end of conversation)
- unknown (You are unsure)

Just respond with the **intent keyword only**, no explanations.
"""
    response = model.invoke(prompt)  # Your LLM
    detected_intent = response.content.strip().lower()
    return detected_intent

# def fast_keyword_intent_detection(message: str) -> Literal["ask_menu", "place_order", "return_deals", "unknown"]:
#     """Fast, cheap keyword-based intent detection."""
#     menu_keywords = ['menu', 'what\'s on the menu', 'today\'s menu']
#     order_keywords = ['order', 'i want', 'i\'d like', 'give me', 'can i get', 'i will take']
#     deal_keywords = ['deal', 'special', 'combo', 'offer', 'promotion']

#     message = message.lower()

#     if any(keyword in message for keyword in deal_keywords):
#         return "return_deals"
#     elif any(keyword in message for keyword in menu_keywords):
#         return "ask_menu"
#     elif any(keyword in message for keyword in order_keywords):
#         return "place_order"
#     else:
#         return "unknown"

# def ask_llm_to_detect_intent(message: str) -> str:
#     """Ask the LLM to classify the intent intelligently."""
#     prompt = f"""
# You are an AI helping a restaurant.

# Classify the customer's intent into one of these:
# - ask_menu
# - place_order
# - return_deals

# User message: "{message}"

# Only respond with the intent name.
# """
#     response = model.invoke(prompt)
#     return response.content.lower()

# def detect_intent(message: str) -> str:
#     """Hybrid detection: fast first, smart fallback."""
#     intent = fast_keyword_intent_detection(message)
#     if intent == "unknown":
#         print("Fast detection failed, asking LLM...")
#         intent = ask_llm_to_detect_intent(message)
#     else:
#         print("Fast detection succeeded!")
#     return intent

# --- Example Testing ---

# test_messages = [
#     "What's on the menu today?",
#     "Can I get a chicken sandwich?",
#     "Do you have any deals?",
#     "Are there any promotions going on?",
#     "I feel like eating something nice.",
#     "Do you have any deals on the menu?",
#     "What kind of deals do you guys have today?"
# ]

# for msg in test_messages:
#     detected_intent = detect_intent(msg)
#     print(f"User: {msg}\nDetected Intent: {detected_intent}\n{'-'*40}")

# Storing PDF in ChromaDB
pdf_reader = PdfReader(PDF_PATH)

pdf_text = pdf_reader.pages[0].extract_text()

def chunk_text(text, max_tokens=252):
    """Chunks text into segments of up to `max_tokens` while respecting sentence boundaries."""
    # Process the text using spaCy
    doc = nlp(text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in doc.sents:
        sentence_tokens = len(sentence.text.split())  # Number of tokens in the sentence
        if current_token_count + sentence_tokens > max_tokens:
            # If adding this sentence would exceed the token limit, start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence.text]
            current_token_count = sentence_tokens
        else:
            # Add sentence to the current chunk
            current_chunk.append(sentence.text)
            current_token_count += sentence_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


chunks = chunk_text(pdf_text)
def storetodb(chunks):
    vectordb = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )

    for idx, chunk in enumerate(chunks):
        vectordb.add_texts(
            texts=[chunk],
            metadatas=[{"chunk_id": idx, "source": "restaurant_menu"}]  # optional but helpful
        )

    vectordb.persist()
    return vectordb

vectordb = storetodb(chunks)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})




def retrieve_from_db(query):
    results = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in results]

# Graph building 

class GraphState(TypedDict):
    user_message: str
    detected_intent: Literal["ask_menu", "return_deals", "place_order", "cancel_order", "goodbye", "unknown"]
    response: str
    chat_history: List[Dict[str, str]] 
    order_list : str

def detect_intent_node(state: GraphState) -> GraphState:
    user_input = input("User: ")
    state["user_message"] = user_input
    intent = detect_intent(state['user_message'], state['chat_history'])
    state['detected_intent'] = intent
    return state

def query_menu_node(state: GraphState) -> GraphState:
    query = state['user_message']
    chat_history = state['chat_history']
    
    # Merge last few messages for context
    history_text = ""
    for turn in chat_history[-3:]:  # You can adjust the number of turns
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    
    # Adjust the prompt to fit the context (ask_menu) with chat history
    retrieved_docs = retrieve_from_db(query)  # <-- Function to retrieve from the database
    context = "\n".join(retrieved_docs)
    
    # Add conversation history to the prompt
    prompt = f"""
    Here is the conversation so far:
    {history_text}
    
    Based on the following menu information:\n{context}
    Answer the user's question:\n{query}
    """
    response = model.invoke(prompt)
    
    updated_chat = chat_history + [{"user": query, "bot": response.content}]
    print(f"Bot: {response.content}")
    state['chat_history'] = updated_chat
    return state


def query_deals_node(state: GraphState) -> GraphState:
    query = state['user_message']
    chat_history = state['chat_history']
    
    # Merge last few messages for context
    history_text = ""
    for turn in chat_history[-3:]:  # Adjust the number of turns as needed
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    
    # Retrieve deals information from the database (or other sources)
    retrieved_docs = retrieve_from_db(query)
    context = "\n".join(retrieved_docs)
    
    # Add conversation history to the prompt
    prompt = f"""
    Here is the conversation so far:
    {history_text}
    
    Based on the following menu information:\n{context}
    Answer the user's question:\n{query}
    """
    response = model.invoke(prompt)
    
    # Update chat history
    updated_chat = chat_history + [{"user": query, "bot": response.content}]
    state['chat_history'] = updated_chat
    print(f"Bot: {response.content}")
    return state


def place_order_node(state: GraphState) -> GraphState:
    query = state['user_message']
    chat_history = state['chat_history']
    history_text = ""
    for turn in chat_history[-5:]:
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    
    retrieved_docs = retrieve_from_db(query)
    context = "\n".join(retrieved_docs)
    prompt = f"""
    The user wants to tell you what they want: {query}
    You are responsible for giving the customer order in the following format:
        - Item 1
          - item 1 price
          - item 1 quantity
        - Item 2
          - item 2 price 
          - item 2 quantity
    You need to replace item name price quantity with the actual prices. You have access to the following information: {context}
    In addition to this information, you also have access to chat history: {history_text}. This means you know the context of the conversation so if user 
    has asked you to add items in the previous messages, make sure to add them as well.
    If there is no chat history, just use {context} to provide answers.
    If the user asked for a deal, the price of the deal is written right next to the deal details
    For example: 
        - 1. Deal 1: 1 Large Pizza + 1.5L Soft Drink - $13.99
    In this case, the price is 13.99
    So incase of deals, you dont need to caluclate deal prices manually.
    Incase of deals, in the output just include deal name and the chosen pizza or burger names
    if the user is asking for a deal. They will also mention which burgers or pizzas they want in that deal. 

    For example a user might say "Alright i will have deal 5 and my chosen pizza is pepperoni!!". This means that they want the deal 5 and in that deal they chose pepperoni. DO NOT CHARGE FOR ANYTHING OTHER THAN THE DEAL.
    NOTE: YOUR OUTPUT SHOULD ONLY CONTAIN THE ORDER IN THE ABOVE MENTIONED FORMAT
    """
    response = model.invoke(prompt)
    
    # Update chat history
    updated_chat = chat_history + [{"user": query, "bot": response.content}]
    state['chat_history'] = updated_chat
    state['order_list'] = response.content
    return state


def handle_unknown_node(state: GraphState) -> GraphState:
    return {"response": "I didn't quite get that. Can you clarify?"}

def cancel_order_node(state: GraphState) -> GraphState:
    return {"response": "No problem! Let me know if you change your mind. 👋"}

def goodbye_node(state: GraphState) -> GraphState:
    return {"response": "Thanks for visiting! Have a great day! 🌟"}

def send_message_func(state: GraphState):
    message = state['order_list']
    pywhatkit.sendwhatmsg("+923304444555",message,23,58)



graph = StateGraph(GraphState)

graph.add_node("detect_intent", detect_intent_node)
graph.add_node("query_menu", query_menu_node)
graph.add_node("query_deals", query_deals_node)
graph.add_node("place_order", place_order_node)
graph.add_node("handle_unknown", handle_unknown_node)
graph.add_node("cancel_order",cancel_order_node)
graph.add_node("goodbye",goodbye_node)
graph.add_node("send_message",send_message_func)

# Define the flow
graph.set_entry_point("detect_intent")

# Routing after intent detection
def routing_function(state):
    """Determine the next node based on the detected intent."""
    intent = state['detected_intent']
    if intent == "ask_menu":
        return "query_menu"
    elif intent == "return_deals":
        return "query_deals"
    elif intent == "place_order":
        return "place_order"
    elif intent == "cancel_order":
        return "cancel_order"
    elif intent == "goodbye":
        return "goodbye"
    else:
        return "handle_unknown"

graph.add_conditional_edges(
    source="detect_intent",
    path=routing_function
)

graph.add_edge("query_menu", "detect_intent")
graph.add_edge("query_deals", "detect_intent")
graph.add_edge("place_order", "send_message")
graph.add_edge("send_message",END)
graph.add_edge("handle_unknown", END)
graph.add_edge("cancel_order",END)
graph.add_edge("goodbye",END)

# Compile the graph!
final_graph = graph.compile()

def printgraph():
    png_graph = final_graph.get_graph().draw_mermaid_png()

    with open("my_graph.png", "wb") as f:
        f.write(png_graph)

printgraph()




def run_bot_conversation():
    state = {
        "user_message": "",
        "detected_intent": "unknown",
        "response": "",
        "chat_history": [],
        "order": {}  # if you have added order tracking
    }

    print("Welcome to the Bot! Type 'exit' to end the conversation.\n")

    #user_input = input("User: ")
    #state["user_message"] = user_input
    
    # Now: let the graph handle everything
    state = final_graph.invoke(state)
    
    print(f"Bot: {state.get('response', 'No response')}")
    
    # Optionally show chat history updates here
    # Update the chat history
    #if "chat_history" in state:
    #    state["chat_history"].append({"user": user_input, "bot": state.get('response', '')})
    
    # How to detect if graph has ended?
    if state.get('__end__', False):
        print("Conversation has ended.")
        return

run_bot_conversation()
