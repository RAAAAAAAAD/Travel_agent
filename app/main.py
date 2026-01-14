import streamlit as st
import os
from dotenv import load_dotenv

# Forza il caricamento del .env dalla cartella radice
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from agent_logic import TravelAgent, TravelProfile
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Travel Agent Pro", layout="wide")

if "agent" not in st.session_state:
    st.session_state.agent = TravelAgent()
if "history" not in st.session_state:
    st.session_state.history = []
if "profile" not in st.session_state:
    st.session_state.profile = TravelProfile()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Profilo Viaggio")
    p = st.session_state.profile
    st.text_input("📍 Partenza", p.partenza, disabled=True)
    st.text_input("🎯 Destinazione", p.destinazione, disabled=True)
    st.text_input("📅 Date", p.date, disabled=True)
    st.text_input("💰 Budget", p.budget, disabled=True)
    st.text_input("🎨 Interessi", p.interessi, disabled=True)
    st.text_input("👥 Gruppo", p.gruppo, disabled=True)
    if st.button("Svuota Chat"):
        st.session_state.history = []
        st.session_state.profile = TravelProfile()
        st.rerun()

# --- CHAT ---
st.title("✈️ Travel Planner AI")

for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Inserisci i dettagli del viaggio..."):
    st.session_state.history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pianificando..."):
            result = st.session_state.agent.run(prompt, st.session_state.history[:-1], st.session_state.profile)
            st.session_state.profile = result["profile"]
            ans = result["messages"][-1].content
            st.markdown(ans)
            st.session_state.history.append(AIMessage(content=ans))
            st.rerun()
