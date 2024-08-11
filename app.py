from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API_KEY")


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API)


index_name="harry-potter-chatbot"

# Create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric='cosine',  
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )



#Loading the index
index = pc.Index(index_name)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# Load the LLM
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 200,
        'temperature': 0.8
    }
)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)