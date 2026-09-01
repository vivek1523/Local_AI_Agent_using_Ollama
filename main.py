from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import Retriever

model = OllamaLLM(model="ithanhloc/llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    Question = input("Ask a question (q to quit): ")
    if Question == "q":
        break

    reviews = Retriever.invoke(Question)
    result = chain.invoke({
        "reviews": reviews,
        "question": Question
    })
    print(result)