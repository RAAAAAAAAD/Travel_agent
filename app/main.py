import streamlit as st
from agent_logic import TravelAgent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

if "agent" not in st.session_state:
    st.session_state.agent = TravelAgent()

if "history" not in st.session_state:
    st.session_state.history = []

if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = "Nessuna informazione raccolta."

# --- SIDEBAR (Pag. 289: Trasparenza e Ispezione) ---
with st.sidebar:
    st.header("🔍 Stato dell'Agente")
    st.markdown("---")
    st.subheader("Informazioni Estratte")
    st.write(st.session_state.extracted_info)
    st.markdown("---")
    if st.button("Reset Conversazione"):
        st.session_state.history = []
        st.session_state.extracted_info = "Nessuna informazione raccolta."
        st.rerun()

# --- MAIN CHAT UI ---
st.title("✈️ AI Travel Agent")
st.caption("Progettato con LangGraph e Llama 3.3 (AI Engineering Cap. 6)")

for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Inserisci i dettagli del tuo viaggio..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("L'agente sta elaborando..."):
            # 1. Esecuzione Logica
            response = st.session_state.agent.run(prompt, st.session_state.history)
            
            # 2. Aggiornamento Memoria Storica
            st.session_state.history.append(HumanMessage(content=prompt))
            st.session_state.history.append(AIMessage(content=response))
            
            # 3. Aggiornamento Sidebar (Estrazione contestuale)
            st.session_state.extracted_info = st.session_state.agent.get_structured_data(st.session_state.history)
            
            st.markdown(response)
            st.rerun()
