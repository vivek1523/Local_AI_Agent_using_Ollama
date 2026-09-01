import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

df = pd.read_csv("realistic_restaurant_reviews.csv")

Embeddings = OllamaEmbeddings(model="mxbai-embed-large")

DB_location = "./Chroma_langchain_DB"
Add_documents = not os.path.exists(DB_location)

if Add_documents:
    Documents = []
    Ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row["Title"] + " " + row["Review"],
            metadata= {
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id=str(i)
        )

        Ids.append(str(i))
        Documents.append(document)

Vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=DB_location,
    embedding_function=Embeddings
)

if Add_documents:
    Vector_store.add_documents(documents=Documents, ids=Ids)

Retriever = Vector_store.as_retriever(
    search_kwargs = {"k" : 5}
)