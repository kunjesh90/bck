#extract_and_store.py

import pickle
import faiss
import numpy as np
from bs4 import BeautifulSoup
from langchain_community.embeddings import SentenceTransformerEmbeddings  # ✅ Updated Import
from langchain_community.docstore.in_memory import InMemoryDocstore  # ✅ Updated Import
from langchain.docstore.document import Document
import os

# Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Read and parse index.html
with open("/Users/anugrah/Desktop/Office Work/website_prnl/my_website/frontend/index.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Extract meaningful content from relevant sections
sections = soup.find_all(["h1", "h2", "h3", "p", "li"])
documents = {}
doc_id = 0

for section in sections:
    text = section.get_text(strip=True)
    if text:
        documents[doc_id] = Document(page_content=text)
        doc_id += 1

# Store in InMemoryDocstore
docstore = InMemoryDocstore(documents)

# Ensure faiss_index directory exists
if not os.path.exists("faiss_index"):
    os.makedirs("faiss_index")  # ✅ Create directory if missing
    print("📂 Created directory: faiss_index")

# ✅ Define FAISS index before writing
dimension = 384  # Adjust based on embedding model
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

# Add documents to FAISS index
for doc_id, document in documents.items():
    embedding = embedding_model.embed_query(document.page_content)
    faiss_index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([doc_id], dtype=np.int64))

# ✅ Save FAISS index and docstore
faiss.write_index(faiss_index, "faiss_index/index.faiss")
with open("docstore.pkl", "wb") as f:
    pickle.dump(docstore, f)

print("✅ Extracted text from index.html and updated FAISS & docstore.pkl!")
