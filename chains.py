from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI  # Uncomment if using OpenAI GPT-4o

# Choose which model to use by toggling the lines below:
# For Gemini (Google):
from dotenv import load_dotenv
import os

load_dotenv()
assert "GOOGLE_API_KEY" in os.environ, "GOOGLE_API_KEY not loaded from .env!"
print("GOOGLE_API_KEY loaded:", os.environ.get("GOOGLE_API_KEY"))


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# For OpenAI GPT-4o, comment out Gemini above and uncomment this:
# llm = ChatOpenAI(model="gpt-4o")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# These chains use the selected LLM (Gemini or OpenAI)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
