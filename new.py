import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Splitting text into small chunks to create embeddings
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


# Handling user questions with prompt technique
def handle_userinput(question):
    # Getting context from previous conversation for better responses
    previous_context = st.session_state.chat_history if st.session_state.chat_history else ""
    
    # Formatting the user question with previous context for a more natural conversation
    prompt = f"""
    Evaluate the following question :
    Question: {question}
    Based on the context i will provide of pdf.
    
    Provide the evaluation in this format but its json format:
    1. **marks**: 
    2. **Feedback**:"""

    # Now pass the prompt into the conversation chain
    response = st.session_state.conversation({"question": prompt})
    st.session_state.chat_history = response['chat_history']
    
    # Display only the answer
    st.write(response['answer'])


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


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                # Convert to chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Generate embeddings
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
