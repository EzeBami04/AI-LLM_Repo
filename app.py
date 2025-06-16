from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import streamlit as st
import os

# ========== Load environment ==========
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "false" 

# ========== Define prompt ==========
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps users in their daily tasks. "
                   "Your default response language is German and English."),
        ("human", "{input}"),
    ]
)

# ========== Setup LLM ==========
llm = Ollama(model="mistral", temperature=0.5)
output_parser = StrOutputParser()

# ========== Setup memory ==========
memory = ConversationBufferMemory(return_messages=True)

# ================== adding buffer memory to the chain =========================#
def format_input_with_history(inputs):
    memory.chat_memory.add_user_message(inputs["input"])
    return {"input": inputs["input"], "history": memory.chat_memory.messages}

chain = RunnableSequence(
    RunnableLambda(format_input_with_history) | prompt | llm | output_parser
)

# ===================== Streamlit UI ==========
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖")
st.title("💬 Langchain + Ollama Chatbot")
st.subheader("Ask me anything (English/German)!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display prior messages
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Get user input via chat
if user_input := st.chat_input("Hi there! Ask your question..."):
    st.chat_message("user").write(user_input)

   
    with st.spinner("Thinking..."):
        response = chain.invoke({"input": user_input})
        st.chat_message("assistant").write(response)

        
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response))
