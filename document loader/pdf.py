from langchain_community.document_loaders import PyPDFLoader

data =PyPDFLoader("document loader/notes.pdf")

docs=data.load()

print(docs[32])