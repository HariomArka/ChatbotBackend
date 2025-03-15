import PyPDF2
import numpy as np
from mistralai import Mistral

MISTRAL_API_KEY = "jxwLtkW5DPDWps5cSroxOx3BogmxOV1L"
client = Mistral(api_key=MISTRAL_API_KEY)

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
    """Generate embeddings for a list of texts using Mistral API."""
    response = client.embeddings.create(model="mistral-embed", inputs=texts)
    embeddings = [item.embedding for item in response.data]
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
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    return response.data[0].embedding

def retrieve_relevant_chunks(query_emb, embeddings, chunks, top_k=3):
    """Retrieve top-k relevant chunks based on query embedding."""
    if not embeddings or not chunks:
        return "No document context available."
    similarities = [cosine_similarity(query_emb, emb) for emb in embeddings]
    indices = np.argsort(similarities)[-top_k:][::-1]
    retrieved = [chunks[i] for i in indices]
    return "\n".join(retrieved)

def generate_answer(messages, model="mistral-large-latest"):
    """Generate an answer using the Mistral chat API."""
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content