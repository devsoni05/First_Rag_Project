import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """you are a helpful ai assistant...
            use only the provided context to answer the question
            if the answer is not present in the document  say : "I could not find the answer in the document."
            """,
        ),
        (
            "human",
            """context:
        {context}
        question :
        {question}
        """,
        ),
    ]
)

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    os.remove(tmp_path)

    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )


# ---------------- Sidebar: PDF upload ----------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
        st.success(f"Loaded: {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.info(f"Current PDF: {st.session_state.pdf_name}")

# ---------------- Chat ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.retriever is None:
    st.info("Upload a PDF from the sidebar to start chatting.")
else:
    query = st.chat_input("Ask something about your PDF...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = template.invoke({"context": context, "question": query})
                response = model.invoke(final_prompt)
                st.markdown(response.content)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
