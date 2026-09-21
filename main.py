from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()



loader = TextLoader("document loader/notes.txt")

docs = loader.load()



template=ChatPromptTemplate.from_messages(
    [
        ("system","you are a ai assistant that summarizes the text"),
        ("human","{data}")
    ]
)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.6
)
promt=template.format_messages(data=docs[0])
result = model.invoke(promt)

print(result.content)