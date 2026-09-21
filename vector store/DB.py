from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Python is a widely used language in artificial intelligence.",
        metadata={"source": "AI_book"},
    ),

    Document(
        page_content="Machine learning allows computers to learn patterns from data without being explicitly programmed.",
        metadata={"source": "ML_book"},
    ),

    Document(
        page_content="Deep learning uses neural networks with multiple layers to solve complex problems such as image recognition and natural language processing.",
        metadata={"source": "DeepLearning_book"},
    ),

    Document(
        page_content="Natural language processing helps computers understand, process, and generate human language.",
        metadata={"source": "NLP_book"},
    ),
]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
# vectors = embeddings.embed_documents(
#     [doc.page_content for doc in docs]
# )


vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is mostly used for machine learning ? ",k=2)

for r in result :
    print(r.page_content)
    print(r.metadata)


retriver=vectorstore.as_retriever()

docs=retriver.invoke("explain deep learning")