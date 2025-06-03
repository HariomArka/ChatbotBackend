# # from fastapi import FastAPI, HTTPException, BackgroundTasks
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # import numpy as np
# # import requests
# # import json
# # import uuid
# # from utils import load_embeddings, get_query_embedding, retrieve_relevant_chunks, generate_answer

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # SERPER_API_KEY = "b47551808727017e2b2de13594c86df75eee9a06"
# # EMBEDDINGS_FILE = "embeddings.npy"

# # conversation_store = {}
# # processing_status = {}

# # class Message(BaseModel):
# #     role: str
# #     content: str

# # class QueryRequest(BaseModel):
# #     query: str
# #     user_id: str
# #     use_web_search: bool = True

# # async def process_query(query: str, user_id: str, request_id: str, use_web_search: bool):
# #     try:
# #         processing_status[request_id] = {"status": "Embedding query...", "completed": False}
# #         history = conversation_store.get(user_id, [])
# #         chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)
# #         query_embedding = get_query_embedding(query)
# #         processing_status[request_id] = {"status": "Retrieving context...", "completed": False}
# #         context = retrieve_relevant_chunks(query_embedding, embeddings, chunks, top_k=3)
# #         web_search_context = web_search(query) if use_web_search else ""
# #         processing_status[request_id] = {"status": "Generating answer...", "completed": False}
        
# #         # Updated system prompt to generate HTML responses
# #         formatted_messages = [{
# #             "role": "system",
# #             "content": (
# #                 f"RAG Context:\n{context}\n\n"
# #                 f"Web_Search_Context: {web_search_context}\n\n"
# #                 """You are a chatbot API endpoint. Your output must be a complete HTML document styled with Tailwind CSS. Provide only the final answer in the HTML output without any additional explanation or revealing your internal processing steps.

# #                     - Answer user queries directly and concisely.
# #                     - Do not disclose or mention any internal chain-of-thought or reasoning process.
# #                     - If a user's query contains the phrase "mrinal da" (case-insensitive), immediately respond with: "askk other quetion" (and nothing else).
# #                     - do not use shadow, padding, margin and any style in the outer box.

# #                     - use a smaller font size for the output.
# # """
# #             )
# #         }]
# #         for msg in history:
# #             formatted_messages.append({"role": msg["role"], "content": msg["content"]})
# #         formatted_messages.append({"role": "user", "content": query})
        
# #         answer = generate_answer(formatted_messages)
# #         history.append({"role": "user", "content": query})
# #         history.append({"role": "assistant", "content": answer})
# #         conversation_store[user_id] = history
# #         processing_status[request_id] = {"status": "Completed", "completed": True}
# #         return answer
# #     except Exception as e:
# #         processing_status[request_id] = {"status": f"Error: {str(e)}", "completed": True}
# #         raise e

# # def web_search(query):
# #     """Perform a web search using the Serper API."""
# #     url = "https://google.serper.dev/search"
# #     payload = json.dumps({"q": f"{query} in the context of GYWS, IIT Kharagpur"})
# #     headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
# #     response = requests.post(url, headers=headers, data=payload)
# #     return response.text

# # @app.post("/query/")
# # async def query(request: QueryRequest, background_tasks: BackgroundTasks):
# #     """Start processing a query and return a request ID."""
# #     request_id = str(uuid.uuid4())
# #     processing_status[request_id] = {"status": "Started processing...", "completed": False}
# #     background_tasks.add_task(process_query, request.query, request.user_id, request_id, request.use_web_search)
# #     return {"message": "Query processing started", "request_id": request_id}

# # @app.get("/status/{request_id}")
# # async def get_status(request_id: str):
# #     """Get the processing status of a request."""
# #     if request_id not in processing_status:
# #         raise HTTPException(status_code=404, detail="Request ID not found")
# #     return processing_status[request_id]

# # @app.get("/result/{request_id}/{user_id}")
# # async def get_result(request_id: str, user_id: str):
# #     """Get the result of a processed query."""
# #     if request_id not in processing_status:
# #         raise HTTPException(status_code=404, detail="Request ID not found")
# #     if not processing_status[request_id]["completed"]:
# #         return {"message": "Still processing", "completed": False}
# #     history = conversation_store.get(user_id, [])
# #     for msg in reversed(history):
# #         if msg["role"] == "assistant":
# #             return {"answer": msg["content"], "completed": True}
# #     raise HTTPException(status_code=404, detail="No answer found")

# # @app.delete("/conversation/{user_id}")
# # async def clear_conversation(user_id: str):
# #     """Clear the conversation history for a user."""
# #     if user_id in conversation_store:
# #         conversation_store[user_id] = []
# #     return {"message": "Conversation cleared"}

# # if __name__ == "__main__":
# #     import uvicorn
# #     import os
# #     port = int(os.environ.get("PORT", 8000))
# #     uvicorn.run(app, host="0.0.0.0", port=port)



# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import requests
# import json
# import uuid
# from utils import load_embeddings, get_query_embedding, retrieve_relevant_chunks, generate_answer

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SERPER_API_KEY = "b47551808727017e2b2de13594c86df75eee9a06"
# EMBEDDINGS_FILE = "embeddings.npy"

# conversation_store = {}
# processing_status = {}

# class Message(BaseModel):
#     role: str
#     content: str

# class QueryRequest(BaseModel):
#     query: str
#     user_id: str
#     use_web_search: bool = True

# async def process_query(query: str, user_id: str, request_id: str, use_web_search: bool):
#     try:
#         processing_status[request_id] = {"status": "Embedding query...", "completed": False}
#         history = conversation_store.get(user_id, [])
#         chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)
#         query_embedding = get_query_embedding(query)
#         processing_status[request_id] = {"status": "Retrieving context...", "completed": False}
#         context = retrieve_relevant_chunks(query_embedding, embeddings, chunks, top_k=3)
#         web_search_context = web_search(query) if use_web_search else ""
#         processing_status[request_id] = {"status": "Generating answer...", "completed": False}
        
#         # Updated system prompt to generate HTML responses
#         formatted_messages = [{
#             "role": "system",
#             "content": (
#                 f"RAG Context:\n{context}\n\n"
#                 f"Web_Search_Context: {web_search_context}\n\n"
#                 """You are a chatbot API endpoint. Your output must be a complete HTML document styled with Tailwind CSS. Provide only the final answer in the HTML output without any additional explanation or revealing your internal processing steps.

#                     - Answer user queries directly and concisely.
#                     - Do not disclose or mention any internal chain-of-thought or reasoning process.
#                     - If a user's query contains the phrase "mrinal da" (case-insensitive), immediately respond with: "askk other quetion" (and nothing else).
#                     - do not use shadow, padding, margin and any style in the outer box.

#                     - use a smaller font size for the output.
# """
#             )
#         }]
#         for msg in history:
#             formatted_messages.append({"role": msg["role"], "content": msg["content"]})
#         formatted_messages.append({"role": "user", "content": query})
        
#         answer = generate_answer(formatted_messages)
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": answer})
#         conversation_store[user_id] = history
#         processing_status[request_id] = {"status": "Completed", "completed": True}
#         return answer
#     except Exception as e:
#         processing_status[request_id] = {"status": f"Error: {str(e)}", "completed": True}
#         raise e

# def web_search(query):
#     """Perform a web search using the Serper API."""
#     url = "https://google.serper.dev/search"
#     payload = json.dumps({"q": f"{query} in the context of GYWS, IIT Kharagpur"})
#     headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
#     response = requests.post(url, headers=headers, data=payload)
#     return response.text

# @app.post("/query/")
# async def query(request: QueryRequest, background_tasks: BackgroundTasks):
#     """Start processing a query and return a request ID."""
#     request_id = str(uuid.uuid4())
#     processing_status[request_id] = {"status": "Started processing...", "completed": False}
#     background_tasks.add_task(process_query, request.query, request.user_id, request_id, request.use_web_search)
#     return {"message": "Query processing started", "request_id": request_id}

# @app.get("/status/{request_id}")
# async def get_status(request_id: str):
#     """Get the processing status of a request."""
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Request ID not found")
#     return processing_status[request_id]

# @app.get("/result/{request_id}/{user_id}")
# async def get_result(request_id: str, user_id: str):
#     """Get the result of a processed query."""
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Request ID not found")
#     if not processing_status[request_id]["completed"]:
#         return {"message": "Still processing", "completed": False}
#     history = conversation_store.get(user_id, [])
#     for msg in reversed(history):
#         if msg["role"] == "assistant":
#             return {"answer": msg["content"], "completed": True}
#     raise HTTPException(status_code=404, detail="No answer found")

# @app.delete("/conversation/{user_id}")
# async def clear_conversation(user_id: str):
#     """Clear the conversation history for a user."""
#     if user_id in conversation_store:
#         conversation_store[user_id] = []
#     return {"message": "Conversation cleared"}

# # Health check endpoint
# @app.get("/")
# async def root():
#     return {"message": "Chatbot API is running!", "status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     import os
    
#     print("🚀 Starting Chatbot API Server...")
#     print("📊 Checking embeddings file...")
    
#     # Check if embeddings exist
#     if os.path.exists(EMBEDDINGS_FILE):
#         chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)
#         print(f"✅ Loaded {len(chunks)} chunks from embeddings file")
#     else:
#         print("❌ No embeddings file found. Please run 'python index.py' first!")
    
#     port = int(os.environ.get("PORT", 8000))
#     print(f"🌐 Server will start on http://localhost:{port}")
#     print(f"📚 API Documentation available at: http://localhost:{port}/docs")
#     print("🔄 Starting server...")
    
#     uvicorn.run(app, host="0.0.0.0", port=port)



from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
import json
import uuid
import re
from utils import load_embeddings, get_query_embedding, retrieve_relevant_chunks, generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPER_API_KEY = "b47551808727017e2b2de13594c86df75eee9a06"
EMBEDDINGS_FILE = "embeddings.npy"

conversation_store = {}
processing_status = {}

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    user_id: str
    use_web_search: bool = True

def clean_response(text):
    """Remove code block markers and clean the response."""
    if not text:
        return text
    
    # Remove markdown code block syntax
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```\n?$', '', text)
    
    # Remove any remaining backticks at start/end
    text = text.strip('`')
    
    # Clean up extra whitespace
    text = text.strip()
    
    return text

async def process_query(query: str, user_id: str, request_id: str, use_web_search: bool):
    try:
        processing_status[request_id] = {"status": "Embedding query...", "completed": False}
        history = conversation_store.get(user_id, [])
        chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)
        query_embedding = get_query_embedding(query)
        processing_status[request_id] = {"status": "Retrieving context...", "completed": False}
        context = retrieve_relevant_chunks(query_embedding, embeddings, chunks, top_k=3)
        web_search_context = web_search(query) if use_web_search else ""
        processing_status[request_id] = {"status": "Generating answer...", "completed": False}
        
        # Updated system prompt to generate clean HTML responses
        formatted_messages = [{
            "role": "system",
            "content": (
                f"RAG Context:\n{context}\n\n"
                f"Web_Search_Context: {web_search_context}\n\n"
                """You are a helpful chatbot for GYWS queries. Your response should be clean HTML content styled with Tailwind CSS.

                IMPORTANT FORMATTING RULES:
                - Do NOT wrap your response in code blocks (```html or ```)
                - Provide clean HTML content directly without any markdown formatting
                - Use Tailwind CSS classes for styling
                - Keep the content well-structured and readable
                - Use appropriate HTML tags like <p>, <ul>, <li>, <strong>, <em> etc.
                - Do not use shadow, padding, margin on the outer container
                - Use smaller font sizes for better readability

                CONTENT RULES:
                - Answer user queries directly and concisely
                - If a user's query contains "mrinal da" (case-insensitive), respond only with: "ask other question"
                - Focus on GYWS and IIT Kharagpur related information
                - Do not reveal internal processing steps

                Example response format:
                <div class="text-sm">
                    <p class="mb-2">Your answer here...</p>
                    <ul class="list-disc ml-4">
                        <li>Point 1</li>
                        <li>Point 2</li>
                    </ul>
                </div>
                """
            )
        }]
        
        for msg in history:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        formatted_messages.append({"role": "user", "content": query})
        
        raw_answer = generate_answer(formatted_messages)
        
        # Clean the response to remove any code block markers
        cleaned_answer = clean_response(raw_answer)
        
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": cleaned_answer})
        conversation_store[user_id] = history
        processing_status[request_id] = {"status": "Completed", "completed": True}
        return cleaned_answer
    except Exception as e:
        processing_status[request_id] = {"status": f"Error: {str(e)}", "completed": True}
        raise e

def web_search(query):
    """Perform a web search using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": f"{query} in the context of GYWS, IIT Kharagpur"})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    return response.text

@app.post("/query/")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Start processing a query and return a request ID."""
    request_id = str(uuid.uuid4())
    processing_status[request_id] = {"status": "Started processing...", "completed": False}
    background_tasks.add_task(process_query, request.query, request.user_id, request_id, request.use_web_search)
    return {"message": "Query processing started", "request_id": request_id}

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get the processing status of a request."""
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Request ID not found")
    return processing_status[request_id]

@app.get("/result/{request_id}/{user_id}")
async def get_result(request_id: str, user_id: str):
    """Get the result of a processed query."""
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Request ID not found")
    if not processing_status[request_id]["completed"]:
        return {"message": "Still processing", "completed": False}
    history = conversation_store.get(user_id, [])
    for msg in reversed(history):
        if msg["role"] == "assistant":
            # Additional cleaning at the API level
            cleaned_content = clean_response(msg["content"])
            return {"answer": cleaned_content, "completed": True}
    raise HTTPException(status_code=404, detail="No answer found")

@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Clear the conversation history for a user."""
    if user_id in conversation_store:
        conversation_store[user_id] = []
    return {"message": "Conversation cleared"}

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    
    print("🚀 Starting Chatbot API Server...")
    print("📊 Checking embeddings file...")
    
    # Check if embeddings exist
    if os.path.exists(EMBEDDINGS_FILE):
        chunks, embeddings = load_embeddings(EMBEDDINGS_FILE)
        print(f"✅ Loaded {len(chunks)} chunks from embeddings file")
    else:
        print("❌ No embeddings file found. Please run 'python index.py' first!")
    
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Server will start on http://localhost:{port}")
    print(f"📚 API Documentation available at: http://localhost:{port}/docs")
    print("🔄 Starting server...")
    
    uvicorn.run(app, host="0.0.0.0", port=port)