from flask import Flask, jsonify, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()



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
    chunks = text_splitter.split_text(text)
    return chunks

# Using Google's embedding004 model to create embeddings and FAISS to store the embeddings
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Handling user input with the prompt technique
def handle_user_input(question):
    previous_context = session_dict.get("chat_history", "")
    prompt = f"""
    Evaluate the following question :
    Question: {question}
    Based on the context I will provide of PDF.
    
    Provide the evaluation in this format:
    1. **marks**: 
    2. **Feedback:"""

    conversation_chain = session_dict.get("conversation")
    if conversation_chain:
        response = conversation_chain({"question": prompt})
        session_dict["chat_history"] = response['chat_history']
        return response['answer']
    else:
        return "Error: No conversation chain initialized."
    
    
# Storing conversations as chain of outputs
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_pdf():
    if not os.path.exists("C:/Users/USER/Desktop/chat_with_doc/class_X_marking _scheme.pdf"):
        return jsonify({"error": "PDF file not found"}), 404

    # Process the PDF
    raw_text = get_pdf_text("C:/Users/USER/Desktop/chat_with_doc/class_X_marking _scheme.pdf")

    # Convert the text to chunks
    text_chunks = get_text_chunks(raw_text)

    # Generate embeddings and create a vector store
    vectorstore = get_vectorstore(text_chunks)

    # Create conversation chain and store it in the session dictionary
    session_dict["conversation"] = get_conversation_chain(vectorstore)
    
with app.app_context():
    process_pdf()
    
# Flask route to handle user questions
@app.route('/evaluate', methods=['POST'])
def ask_question():
    
    data = request.get_json()
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]
    answer = handle_user_input(question)
    print(answer)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)
