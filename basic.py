from typing import List, Sequence
from dotenv import load_dotenv
import os
from pydantic import SecretStr
from chains import generation_chain, reflection_chain

# Load .env variables
load_dotenv()

GOOGLE_API_KEY = SecretStr(os.environ["GOOGLE_API_KEY"])
api_key_str = GOOGLE_API_KEY.get_secret_value()


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


REFLECT = "reflect"
GENERATE = "generate"
graph = MessageGraph()

def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })


def reflect_node(messages):
    response = reflection_chain.invoke({
        "messages": messages
    })
    return [HumanMessage(content=response.content)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)


def should_continue(state):
    if (len(state) > 6):
        return END 
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

print(response)
