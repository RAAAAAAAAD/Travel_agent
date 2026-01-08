import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def verify():
    key = os.getenv("GROQ_API_KEY")
    if not key or "inserisci" in key:
        print("❌ Errore: Configura la GROQ_API_KEY nel file .env")
        return
    try:
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=key)
        res = llm.invoke("Ping")
        print(f"✅ Connessione Groq OK! Modello: llama3-70b-8192")
    except Exception as e:
        print(f"❌ Errore API: {e}")

if __name__ == "__main__":
    verify()
