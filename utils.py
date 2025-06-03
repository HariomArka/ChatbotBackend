# # import PyPDF2
# # import numpy as np
# # from mistralai import Mistral

# # MISTRAL_API_KEY = "jxwLtkW5DPDWps5cSroxOx3BogmxOV1L"
# # client = Mistral(api_key=MISTRAL_API_KEY)

# # def extract_text_from_pdf(pdf_path):
# #     """Extract text from a PDF file."""
# #     reader = PyPDF2.PdfReader(pdf_path)
# #     text = ""
# #     for page in reader.pages:
# #         extracted = page.extract_text()
# #         if extracted:
# #             text += extracted + "\n"
# #     return text

# # def chunk_text(text, chunk_size=50):
# #     """Split text into chunks of specified word size."""
# #     words = text.split()
# #     chunks = []
# #     for i in range(0, len(words), chunk_size):
# #         chunk = " ".join(words[i:i + chunk_size])
# #         chunks.append(chunk)
# #     return chunks

# # def get_embeddings(texts):
# #     """Generate embeddings for a list of texts using Mistral API."""
# #     response = client.embeddings.create(model="mistral-embed", inputs=texts)
# #     embeddings = [item.embedding for item in response.data]
# #     return embeddings

# # def load_embeddings(npy_file):
# #     """Load chunks and embeddings from an NPY file."""
# #     try:
# #         data = np.load(npy_file, allow_pickle=True).item()
# #         return data["chunks"], data["embeddings"]
# #     except (FileNotFoundError, ValueError):
# #         return [], []

# # def cosine_similarity(a, b):
# #     """Calculate cosine similarity between two vectors."""
# #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # def get_query_embedding(query):
# #     """Generate embedding for a single query text."""
# #     response = client.embeddings.create(model="mistral-embed", inputs=[query])
# #     return response.data[0].embedding

# # def retrieve_relevant_chunks(query_emb, embeddings, chunks, top_k=3):
# #     """Retrieve top-k relevant chunks based on query embedding."""
# #     if not embeddings or not chunks:
# #         return "No document context available."
# #     similarities = [cosine_similarity(query_emb, emb) for emb in embeddings]
# #     indices = np.argsort(similarities)[-top_k:][::-1]
# #     retrieved = [chunks[i] for i in indices]
# #     return "\n".join(retrieved)

# # def generate_answer(messages, model="mistral-large-latest"):
# #     """Generate an answer using the Mistral chat API."""
# #     chat_response = client.chat.complete(model=model, messages=messages)
# #     return chat_response.choices[0].message.content



# import PyPDF2
# import numpy as np
# import google.generativeai as genai

# # Replace with your Gemini API key
# GEMINI_API_KEY = "AIzaSyDZlILBNwBxhKWhiMULBAJs1XIaDl4IBnY"
# genai.configure(api_key=GEMINI_API_KEY)

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     reader = PyPDF2.PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         extracted = page.extract_text()
#         if extracted:
#             text += extracted + "\n"
#     return text

# def chunk_text(text, chunk_size=50):
#     """Split text into chunks of specified word size."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks

# def get_embeddings(texts):
#     """Generate embeddings for a list of texts using Gemini API."""
#     embeddings = []
#     for text in texts:
#         try:
#             result = genai.embed_content(
#                 model="models/embedding-001",
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             embeddings.append(result['embedding'])
#         except Exception as e:
#             print(f"Error generating embedding for text: {e}")
#             # Fallback: create a zero vector of appropriate dimension
#             embeddings.append([0.0] * 768)  # embedding-001 uses 768 dimensions
#     return embeddings

# def load_embeddings(npy_file):
#     """Load chunks and embeddings from an NPY file."""
#     try:
#         data = np.load(npy_file, allow_pickle=True).item()
#         return data["chunks"], data["embeddings"]
#     except (FileNotFoundError, ValueError):
#         return [], []

# def cosine_similarity(a, b):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def get_query_embedding(query):
#     """Generate embedding for a single query text."""
#     try:
#         result = genai.embed_content(
#             model="models/embedding-001",
#             content=query,
#             task_type="retrieval_query"
#         )
#         return result['embedding']
#     except Exception as e:
#         print(f"Error generating query embedding: {e}")
#         return [0.0] * 768  # Fallback zero vector

# def retrieve_relevant_chunks(query_emb, embeddings, chunks, top_k=3):
#     """Retrieve top-k relevant chunks based on query embedding."""
#     if not embeddings or not chunks:
#         return "No document context available."
#     similarities = [cosine_similarity(query_emb, emb) for emb in embeddings]
#     indices = np.argsort(similarities)[-top_k:][::-1]
#     retrieved = [chunks[i] for i in indices]
#     return "\n".join(retrieved)

# def generate_answer(messages, model="gemini-1.5-flash"):
#     """Generate an answer using the Gemini chat API."""
#     try:
#         # Convert messages format to work with Gemini
#         gemini_model = genai.GenerativeModel(model)
        
#         # Extract system prompt and conversation history
#         system_prompt = ""
#         conversation_history = []
        
#         for msg in messages:
#             if msg["role"] == "system":
#                 system_prompt = msg["content"]
#             elif msg["role"] == "user":
#                 conversation_history.append(f"User: {msg['content']}")
#             elif msg["role"] == "assistant":
#                 conversation_history.append(f"Assistant: {msg['content']}")
        
#         # Combine system prompt with conversation
#         full_prompt = f"{system_prompt}\n\nConversation:\n" + "\n".join(conversation_history)
        
#         response = gemini_model.generate_content(full_prompt)
#         return response.text
        
#     except Exception as e:
#         print(f"Error generating answer: {e}")
#         return "I apologize, but I'm having trouble generating a response right now. Please try again later."


import PyPDF2
import numpy as np
import google.generativeai as genai
import re

# Replace with your Gemini API key
GEMINI_API_KEY = "AIzaSyDZlILBNwBxhKWhiMULBAJs1XIaDl4IBnY"
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text, chunk_size=50):
    """Split text into chunks of specified word size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embeddings(texts):
    """Generate embeddings for a list of texts using Gemini API."""
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            # Fallback: create a zero vector of appropriate dimension
            embeddings.append([0.0] * 768)  # embedding-001 uses 768 dimensions
    return embeddings

def load_embeddings(npy_file):
    """Load chunks and embeddings from an NPY file."""
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        return data["chunks"], data["embeddings"]
    except (FileNotFoundError, ValueError):
        return [], []

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_query_embedding(query):
    """Generate embedding for a single query text."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return [0.0] * 768  # Fallback zero vector

def retrieve_relevant_chunks(query_emb, embeddings, chunks, top_k=3):
    """Retrieve top-k relevant chunks based on query embedding."""
    if not embeddings or not chunks:
        return "No document context available."
    similarities = [cosine_similarity(query_emb, emb) for emb in embeddings]
    indices = np.argsort(similarities)[-top_k:][::-1]
    retrieved = [chunks[i] for i in indices]
    return "\n".join(retrieved)

def clean_html_response(text):
    """Clean and validate HTML response."""
    if not text:
        return text
    
    # Remove any markdown code block syntax
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```\n?$', '', text)
    
    # Remove any remaining backticks
    text = text.strip('`')
    
    # Clean up extra whitespace but preserve HTML structure
    text = text.strip()
    
    return text

def generate_answer(messages, model="gemini-1.5-flash"):
    """Generate an answer using the Gemini chat API."""
    try:
        # Convert messages format to work with Gemini
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Extract system prompt and conversation history
        system_prompt = ""
        conversation_history = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation_history.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation_history.append(f"Assistant: {msg['content']}")
        
        # Combine system prompt with conversation
        full_prompt = f"{system_prompt}\n\nConversation:\n" + "\n".join(conversation_history)
        
        # Add explicit instruction to avoid code blocks
        full_prompt += "\n\nIMPORTANT: Respond with clean HTML content only. Do NOT use code blocks (```html) or any markdown formatting. Provide the HTML directly."
        
        response = gemini_model.generate_content(full_prompt)
        
        # Clean the response
        cleaned_response = clean_html_response(response.text)
        
        return cleaned_response
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return '<div class="text-sm text-red-600"><p>I apologize, but I\'m having trouble generating a response right now. Please try again later.</p></div>'