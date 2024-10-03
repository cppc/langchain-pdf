import os
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"),
    embeddings
)


def build_retriever(chat_args, k):
    search_kwargs = {
        "filter": {
            "pdf_id": chat_args.pdf_id
        },
        "k": k
    }
    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )