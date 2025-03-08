import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    
    st.title("📚 AI Document Assistant")
    
    menu = ["Login", "Chat with Document", "Check Embeddings", "Upload PDF"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.subheader("User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            st.success("Logged in successfully!")
    
    elif choice == "Chat with Document":
        st.subheader("Chat with Your PDFs")
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        user_question = st.text_input("Ask a question about your documents:")
        
        if user_question and st.session_state.conversation:
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history.append((user_question, response['answer']))
        
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
    
    elif choice == "Check Embeddings":
        st.subheader("Embedding Information")
        st.write("View the stored embeddings and related metadata.")
    
    elif choice == "Upload PDF":
        st.subheader("Upload and Process PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now chat with your documents.")

if __name__ == '__main__':
    main()