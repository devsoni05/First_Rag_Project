from dotenv import load_dotenv
from langchain_groq import ChatGroq

# from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_text_splitters import TokenTextSplitter
load_dotenv()


# loader = PyPDFLoader("document loader/notes.pdf")

# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma-db", embedding_function=embeddings)


retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)
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

print("----------------enter 0 to exit-------------")

while True:
    query = input("you :  ")
    if query == "0":
        break
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_promt = template.invoke({"context": context, "question": query})
    response = model.invoke(final_promt)

    print(f"\n AI : {response.content}")
