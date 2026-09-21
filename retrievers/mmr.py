from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

docs = [
    Document(
        page_content="gradient descent is an optimization algorithm used in machine learning"
    ),
    Document(
        page_content="gradient descent minimizes the cost function by updating model parameters in the direction of the negative gradient"
    ),
    Document(
        page_content="the learning rate controls the size of each step taken during gradient descent"
    ),
    Document(
        page_content="gradient descent repeatedly calculates the gradient and updates the weights until the cost function reaches a minimum"
    ),
    Document(
        page_content="a smaller learning rate makes gradient descent converge slowly, while a very large learning rate can cause the algorithm to overshoot the minimum"
    ),
]

embeddings = HuggingFaceBgeEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


print("=============similarity search results===========")
similarity_docs = similarity_retriever.invoke("what is gradient descent ")
for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

print("\n=============MMR results===========\n")

mmr_docs = mmr_retriever.invoke("what is gradient descent")

for doc in mmr_docs:
    print(doc.page_content)
