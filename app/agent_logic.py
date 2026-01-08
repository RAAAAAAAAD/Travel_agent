import os
from typing import TypedDict, Annotated, List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# Tool di ricerca robusto (Pag. 279)
def travel_web_search(query: str):
    try:
        with DDGS() as ddgs:
            # Query pulita per evitare brand irrilevanti
            clean_query = f"{query} -shoes -scarpe -zalando -ai -software"
            results = []
            # Forziamo ricerca italiana per prezzi in Euro
            for r in ddgs.text(clean_query, region='it-it', max_results=5):
                results.append(f"INFO: {r['title']} | DESC: {r['body']} | URL: {r['href']}")
            return "\n\n".join(results) if results else "Nessun dato web trovato."
    except Exception:
        return "Errore connessione ricerca."

# Definizione dello Stato Strutturato (Pag. 51: Structural Integrity)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_data: Dict[str, str] # Qui salviamo i dati estratti
    next_node: str

class TravelAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, # Obbligatorio 0 per precisione
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        workflow = StateGraph(AgentState)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("architect", self._architect_node)
        
        workflow.set_entry_point("analyzer")
        workflow.add_conditional_edges("analyzer", lambda x: x["next_node"], {"research": "researcher", "ask": END})
        workflow.add_edge("researcher", "architect")
        workflow.add_edge("architect", END)
        
        self.app = workflow.compile()

    def _analyzer_node(self, state: AgentState):
        # Nodo 1: Estrazione dati (Pag. 270: Query Rewriting)
        prompt = """Analizza la chat. Estrai queste info in formato testo: 
        Partenza, Destinazione, Date, Budget, Interessi, Gruppo.
        Se mancano info, chiedile. Se le hai TUTTE, scrivi 'READY_TO_SEARCH' e l'elenco dei dati."""
        
        res = self.llm.invoke([HumanMessage(content=prompt)] + state['messages'])
        is_ready = "READY_TO_SEARCH" in res.content
        return {
            "messages": [res],
            "next_node": "research" if is_ready else "ask",
            "user_data": {"summary": res.content} if is_ready else {}
        }

    def _research_node(self, state: AgentState):
        # Nodo 2: Ricerca Web focalizzata sui dati estratti
        summary = state["user_data"]["summary"]
        q_flights = f"voli aerei reali prezzi e compagnie per {summary}"
        q_hotels = f"nomi hotel reali e prezzi notte per {summary}"
        
        return {
            "user_data": {
                "flights": travel_web_search(q_flights),
                "hotels": travel_web_search(q_hotels),
                "summary": summary
            }
        }

    def _architect_node(self, state: AgentState):
        # Nodo 3: Generazione Itinerario con Grounding Totale (Pag. 254)
        data = state["user_data"]
        
        prompt = f"""
        SEI UN AGENTE DI VIAGGIO. DEVI RISPONDERE ESCLUSIVAMENTE PER QUESTA RICHIESTA:
        DATI UTENTE: {data['summary']}
        
        DATI RICERCA WEB (VOLI): {data['flights']}
        DATI RICERCA WEB (HOTEL): {data['hotels']}
        
        REGOLE TASSATIVE:
        1. La destinazione DEVE essere quella richiesta ({data['summary']}). NON PROPORRE ROMA.
        2. Il budget totale NON DEVE superare quello dell'utente.
        3. Se i dati web sono scarsi, usa la tua conoscenza per suggerire compagnie (es. AirBaltic, Ryanair) 
           e hotel reali a Vilnius (es. Radisson Blu, Hotel Tilto) che rientrano nel budget di 400€.
        4. Elenca NOMI HOTEL, VOLI e ITINERARIO GIORNO PER GIORNO.
        """
        res = self.llm.invoke(prompt)
        return {"messages": [res]}

    def run(self, user_input: str, history: list):
        inputs = {"messages": history + [HumanMessage(content=user_input)], "user_data": {}, "next_node": ""}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content
