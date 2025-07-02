from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate

# Use the HF transformers pipeline directly
hf_pipe = pipeline(
    model="Menlo/Jan-nano",
    task="text-generation",  # okay here
    max_new_tokens=100,
    temperature=0.5
)

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Directly invoke the model
response = llm.invoke("What is the capital of France?")
print(response)
