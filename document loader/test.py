from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter

loader = TextLoader("document loader/notes.txt")

data = loader.load()

# character splitting or character chunking
# text_splitter = CharacterTextSplitter(separator="", chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_documents(data)
# print(len(texts))


# token based splitting


text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

texts = text_splitter.split_documents(data)

print(len(texts))



#  recursive character splitting --->mosrly used