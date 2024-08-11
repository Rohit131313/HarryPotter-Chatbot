from src.helper import text_split, download_hugging_face_embeddings
from pinecone import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from langchain.schema import Document

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API_KEY")

with open("Harry_Potter_all_books_preprocessed.txt") as f:
    data = f.read()

data = [Document(page_content=data)]

text_chunks = text_split(data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API)


index_name="harry-potter-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

vectorstore.add_texts([t.page_content for t in text_chunks])