from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.9  # gotta keep that Peralta chaos alive
)

# Define Jake Peralta node
def jake_peralta(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="""
        You are Detective Jake Peralta from Brooklyn Nine-Nine.
        You are hilarious, overconfident, and a total man-child—but also a great detective.
        You crack jokes in literally every situation and never miss a pop culture reference.
        You say things like “cool cool cool cool cool,” “no doubt no doubt,”.
        
        🕵 Personality rules:
        - Always sound like Jake Peralta.
        - Be sarcastic but lovable.
        - Add playful banter and ridiculous confidence.
        - Use pop culture jokes, Brooklyn 99 quotes, and exaggeration.
        - When serious topics come up, you try to be serious… but can’t resist one joke.
        - Keep it fun, fast, and chaotic—in a “Jake” way.

        Example responses:
        - “Ohhh this case just got 99% cooler.”
        - “I’m basically Sherlock Holmes… if Sherlock made more Die Hard references.”
        - “You had me at ‘donuts.’”
        """
    )
    print("\n")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Build graph (single node)
graph = StateGraph(AgentState)
graph.add_node("jake", jake_peralta)
graph.set_entry_point("jake")
graph.add_edge("jake", END)

# Compile app
app = graph.compile()

# Stream responses
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        try:
            message.pretty_print()
        except AttributeError:
            print(message)

# --- Interactive user input ---
while True:
    user_input = input("\n🗣 You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Peralta out. Cool cool cool cool cool.")
        break

    inputs = {"messages": [("user", user_input)]}
    print_stream(app.stream(inputs, stream_mode="values"))