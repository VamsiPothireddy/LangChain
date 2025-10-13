from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv

# Load your .env file
load_dotenv(dotenv_path="/Users/Vamsi/git/LangChain/.env")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",  # <-- must match model's supported task
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template= "Tell me about {person}.",
    input_variables=['person']
)
prompt2 = PromptTemplate(
    template="Check details about below person and tell me his age?\n {text} ",
      input_variables=['text'])

parser = StrOutputParser()

# result= model.invoke (prompt1.invoke({'person':'Virat Kohli'}))
# print(result.content)


# result2= model.invoke (prompt2.invoke({}))
# print("result conetent is ",result2.content)

chain = prompt1 | model | parser | prompt2 | model | parser

chain_result = chain.invoke({'person': 'Virat Kohli'})
print(chain_result)

chain.get_graph().print_ascii()