from flask import Flask, jsonify, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import re
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name
index_name = "bharatomr1"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

# Instantiate the index
index = pc.Index(index_name)
# Initialize a global dictionary to simulate session-like behavior
session_dict = {}

# Function to extract text from PDFs
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Splitting text into smaller chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Using Google's embedding model and Pinecone to create a vector store
def get_vectorstore(text_chunks, index):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.vectorstores import Pinecone

    # Initialize embedding model
    embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Create embeddings for the text chunks
    chunk_embeddings = [embed.embed_query(chunk) for chunk in text_chunks]

    # Prepare data for upsert
    data = [(str(i), chunk_embeddings[i], {"text": text_chunks[i]}) for i in range(len(text_chunks))]

    # Upsert data into the Pinecone index
    index.upsert(vectors=data)

    # Initialize Pinecone vector store for LangChain
    text_field = "text"
    vectorstore = Pinecone(index, embed.embed_query, text_field)
    return vectorstore

# Handling user input with a prompt
def handle_user_input(user_answer, max_marks):
    prompt = f"""
    Evaluate the following response based on the given context:

    
    User Answer: {user_answer}
    Maximum Marks: {max_marks}

    Evaluate the correctness, completeness, and relevance of the answer.  
    Award marks out of {max_marks} based on the following:
    - Full marks for a complete and accurate answer.
    - Deduct marks proportionally for incomplete or partially correct answers.
    
    evaluation strictly in this format:
    1. **Marks**: [Insert marks out of {max_marks}]
    2. **Feedback**: [Provide a 3-4 sentence evaluation of the answer.]
    """

    conversation_chain = session_dict.get("conversation")
    if conversation_chain:
        response = conversation_chain({"question": prompt})
        session_dict["chat_history"] = response["chat_history"]
        return response["answer"]
    else:
        return "Error: No conversation chain initialized."

# Create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Process the PDF and set up the vector store
def process_all_pdfs():
    cwd = os.getcwd()
    pdf_files = [f for f in os.listdir(cwd) if f.endswith('.pdf')]
    
    if not pdf_files:
        return jsonify({"error": "No PDF files found in the current directory"}), 404

    all_text_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(cwd, pdf_file)
        print(f"Processing PDF: {pdf_file}")
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        all_text_chunks.extend(text_chunks)

    vectorstore = get_vectorstore(all_text_chunks, index)
    session_dict["conversation"] = get_conversation_chain(vectorstore)

    return {"message": "All PDFs processed successfully", "processed_files": pdf_files}

# Process PDFs at startup
with app.app_context():
    result = process_all_pdfs()
    print(result)

# API Endpoint
@app.route('/evaluate', methods=['POST'])
def ask_question():
    data = request.get_json()
    if "user_answer" not in data or "max_marks" not in data:
        return jsonify({"error": "Missing 'user_answer', 'max_marks' in the request body"}), 400

    question =data["user_answer"]
    max_marks = data["max_marks"]

    answer = handle_user_input(question, max_marks)
    print(answer)

    score_match = re.search(r"\*\*Marks\*\*:\s*([0-9]*\.?[0-9]+)", answer)
    feedback_match = re.search(r"\*\*Feedback\*\*:\s*(.+)", answer, re.DOTALL)

    return jsonify({
        "Marks": score_match.group(1) if score_match else "No score found",
        "Feedback": feedback_match.group(1).strip() if feedback_match else "No feedback found"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
