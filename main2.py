from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


url = "https://jevaraat-1.onrender.com/"
loader = WebBaseLoader(url)

docs = loader.load()


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "just  summarize the given website in four to five lines max to max...",
        ),
        ("human", "{data}"),
    ]
)
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6)
promt = template.format_messages(data=docs)
result = model.invoke(promt)

print(result.content)
