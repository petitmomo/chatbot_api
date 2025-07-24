from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

import uvicorn
app = FastAPI()

TOGETHER_API_KEY = "together_api_key"
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
chat_history = []

class Message(BaseModel):
    message: str

@app.post("/chat")
def chat(msg: Message):
    chat_history.append({"role": "user", "content": msg.message})
    prompt = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in chat_history)
    prompt += "<|assistant|>"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.xyz/v1/completions", json=payload, headers=headers)

    if response.status_code == 200:
        assistant_reply = response.json()["choices"][0]["text"].strip()
        chat_history.append({"role": "assistant", "content": assistant_reply})
        return {"response": assistant_reply}
    else:
        return {"error": response.text}

@app.get("/history")
def get_history():
    return {"history": chat_history}

# RAG
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb_path = "./vectordb"

@app.post("/upload-pdf")
def upload_pdfs(files: List[UploadFile] = File(...)):
    all_pages = []
    for file in files:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        loader = PyPDFLoader(temp_path)
        pages = loader.load_and_split()
        all_pages.extend(pages)

        os.remove(temp_path)

    vectordb = Chroma.from_documents(
        all_pages,
        embedding=embedding_model,
        persist_directory=vectordb_path
    )
    vectordb.persist()

    return {"status": f"{len(files)} fichiers PDF indexés avec succès."}


@app.post("/ask-doc")
def ask_doc(query: Message):
    # Recharger la base vectorielle
    vectordb = Chroma(persist_directory=vectordb_path, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()

    # Obtenir le contexte depuis les documents
    context_docs = retriever.get_relevant_documents(query.message)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # Construire le prompt pour Together API
    prompt = f"""Tu es un assistant intelligent. Tu dois répondre à la question de l'utilisateur en te basant uniquement sur les documents suivants :

{context_text}

Question : {query.message}
Réponse :"""

    # Requête vers Together API
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.xyz/v1/completions", json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()["choices"][0]["text"].strip()
        return {"response": result}
    else:
        return {"error": response.text}
    
