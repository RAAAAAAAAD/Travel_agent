import streamlit as st
from agent_logic import TravelAgent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Travel Agent Pro", page_icon="✈️")

if "agent" not in st.session_state:
    st.session_state.agent = TravelAgent()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("✈️ Travel Agent AI")
st.caption("Ricerca Web in tempo reale per Hotel e Voli reali")

# Chat UI
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ciao! Da dove vorresti partire?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Ragionando..."):
            response = st.session_state.agent.run(prompt, st.session_state.history)
            st.markdown(response)
    
    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=response))
