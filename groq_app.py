from langchain_groq import ChatGroq
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import subprocess
import multiprocessing
import uvicorn


from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import streamlit as st
from langserve import add_routes
import os
load_dotenv()
# ========== Load environment ==========
groq_api_key = os.getenv("groq_api")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

#===========================================================
#====================Define prompts==========================

prompt = ChatPromptTemplate.from_messages(
    [("system",  "You are a helpful assistant that helps users in their daily tasks."),
     ("user", "{input}")
     ]
     )

#====================LLM ====================================
llm = ChatGroq(model='gemma2-9b-it', temperature=0.5, groq_api_key=groq_api_key)

#==================Memory===========================
memory = ConversationBufferMemory(return_messages=True)
def format_input_with_history(inputs):
    memory.chat_memory.add_user_message(inputs["input"])
    return {"input": inputs["input"], "history": memory.chat_memory.messages}

lambda_ =  RunnableLambda(format_input_with_history)
#===================== output parser =========================
parser = StrOutputParser()

#======================== Chain ================================
chain = RunnableSequence(
    lambda_ | prompt | llm | parser
    )

app =  FastAPI(title="Langchain Groq Chatbot",
               description="Langchain Groq Chatbot using Groq LLM",
               version="0.1.0")
#===================== Add routes =========================
add_routes(app, chain, 
           path="/groq")

#===================== Streamlit UI =========================
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖")
st.title("💬 quantum chatbot")
st.subheader("Hi there I am quantum your personal assistant. Ask me anything!")

# ===================Initialize session state for chat history================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#===================== Display prior messages =========================
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# ===================== Get user input via chat =========================
if user_input := st.chat_input("Hi there! Ask your question..."):
    st.chat_message("user").write(user_input)

   
    with st.spinner("Thinking..."):
        response = chain.invoke({"input": user_input})
        st.chat_message("assistant").write(response)

        
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response))

def run_fastapi():
    uvicorn.run("groq_app", host="0.0.0.0", port=8020, reload=False)

def run_streamlit():
    subprocess.run([
        "streamlit", "run", "groq_app.py",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_fastapi)
    p2 = multiprocessing.Process(target=run_streamlit)

    p1.start()
    p2.start()

    p1.join()
    p2.join()   