import os
import numpy as np
from utils import extract_text_from_pdf, chunk_text, get_embeddings

UPLOADED_PDF_PATH = "uploads/"
EMBEDDINGS_FILE = "embeddings.npy"

def index_pdfs():
    """Index all PDFs in the uploads folder and save embeddings."""
    all_chunks = []
    for filename in os.listdir(UPLOADED_PDF_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(UPLOADED_PDF_PATH, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
    if all_chunks:
        embeddings = get_embeddings(all_chunks)
        data = {"chunks": all_chunks, "embeddings": embeddings}
        np.save(EMBEDDINGS_FILE, data, allow_pickle=True)
        print(f"Indexed {len(all_chunks)} chunks from {len(os.listdir(UPLOADED_PDF_PATH))} PDFs.")
    else:
        print("No PDFs found in upload folder.")

if __name__ == "__main__":
    os.makedirs(UPLOADED_PDF_PATH, exist_ok=True)  # Ensure uploads folder exists
    index_pdfs()