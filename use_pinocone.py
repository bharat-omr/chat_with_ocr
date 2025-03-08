from flask import Flask, jsonify, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
import os
import re
from pinecone import Pinecone, ServerlessSpec
import pinecone  

app = Flask(__name__)

# Load environment variables
load_dotenv()

from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # Use a default value

pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name
index_name = "bharatomr1"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# Create the index if it doesn't exist
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
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

    vectorstore = PineconeVectorStore(index=index, embedding=embed)
    return vectorstore

# Handling user input with a prompt
def handle_user_input(question):
    prompt = f"""
    Evaluate the following {question} based on the given context:

    Answer: {question}. Do not draw any inferences from the user's answer, and do not imply anything. 
    Please note:
    - The question may contain minor issues in the answer, so you should adjust the marks accordingly.
    - If the answer is partial, award half marks or an appropriate fraction based on the context and the provided marks system.
    - Provide clear feedback based on the answer, without adding extra explanations beyond the response given.

    Here is an example of the expected output format:
    {{
      "Marks": "10",
      "Feedback": "this answer is good but need some minor changes."
    }}
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
    # Get the current working directory
    cwd = os.getcwd()
    
    # Find all PDF files in the directory
    pdf_files = [f for f in os.listdir(cwd) if f.endswith('.pdf')]
    
    if not pdf_files:
        return jsonify({"error": "No PDF files found in the current directory"}), 404

    # Initialize a list to store all text chunks
    all_text_chunks = []

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(cwd, pdf_file)
        print(f"Processing PDF: {pdf_file}")
        raw_text = get_pdf_text(pdf_path)  # Extract text
        text_chunks = get_text_chunks(raw_text)  # Split text into chunks
        all_text_chunks.extend(text_chunks)  # Append chunks to the list

    # Generate embeddings for all chunks and create a unified vector store
    vectorstore = get_vectorstore(all_text_chunks,index)

    # Create conversation chain and store it in the session dictionary
    session_dict["conversation"] = get_conversation_chain(vectorstore)

    return {"message": "All PDFs processed successfully", "processed_files": pdf_files}
with app.app_context():
    result = process_all_pdfs()
    print(result)    
@app.route('/evaluate', methods=['POST'])
def ask_question():
    data = request.get_json()
    if "user_answer" not in data:
        return jsonify({"error": "No user_answer provided"}), 400

    question = data["user_answer"]
    answer = handle_user_input(question)
    print(answer)
    score_match = re.search(r'"Marks":\s*"([\d\.\/]+)"', answer)
    feedback_match = re.search(r'"Feedback":\s*"(.+?)"', answer, re.DOTALL)

    return jsonify({
        "Marks": score_match.group(1) if score_match else "No score found",
        "Feedback": feedback_match.group(1).strip() if feedback_match else "No feedback found"
    })

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=9000, debug=True)
