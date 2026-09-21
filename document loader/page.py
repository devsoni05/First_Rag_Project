from langchain_community.document_loaders import WebBaseLoader

url = "https://jevaraat-1.onrender.com/"
loader = WebBaseLoader(url)

docs = loader.load()

print(docs)