import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv

from typing import TypedDict,Annotated,Literal
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

def get_chat():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        temperature=0,
    )
    return ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
    response: Annotated[list[BaseMessage],add_messages]
    
def chat_node(state: ChatState) :
    messages = state["response"]
    chat = get_chat().invoke(messages)
    return {"response": [chat]}


checkpoint=MemorySaver()
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpoint)



#not maintaining state because we dont want to do it manually we want the chatbot workflow state to be able to handle teh memory of the converstaion
thread_id='1'
#. thraed id is requires to maitain state of that converstaion for a particular user
# while True:
#     user_input = input("User: ")
#     config = {'configurable': {'thread_id': thread_id}}
#     response = chatbot.invoke({"response": [HumanMessage(content=user_input)]}, config=config)
#     print("AI:", response["response"][-1].content)
        
# for message_chunk, metadata in chatbot.stream({"response": [HumanMessage(content="write 500 word essay on pasta")]}, config={'configurable': {'thread_id': thread_id}},stream_mode="messages"):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)    
    
    

    
