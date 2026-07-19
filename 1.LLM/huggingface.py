from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Dummy commit 6
# Load your .env file
load_dotenv(dotenv_path="/Users/Vamsi/git/LangChain/.env")

print("Token visible?", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

# ✅ Use the correct task
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",  # <-- must match model's supported task
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(
    llm=llm)



prompt1 = PromptTemplate(
    template=" Tell me about {person}.",
    input_variables=["person"]
)

prompt2 = PromptTemplate(
    template=" Write 5 line summary on fallowing description {text}.",
    input_variables=["text"]
)
parser = StrOutputParser()

chain =prompt1 | model | parser|prompt2 | model| parser

result = chain.invoke({"person": "Albert Einstein"})
print(result)
