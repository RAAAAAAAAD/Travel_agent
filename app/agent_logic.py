import os
from typing import TypedDict, Annotated, List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()

def travel_web_search(query: str):
    try:
        with DDGS() as ddgs:
            clean_query = f"{query} -shoes -scarpe -zalando -ai -software"
            results = []
            for r in ddgs.text(clean_query, region='it-it', max_results=5):
                results.append(f"INFO: {r['title']} | DESC: {r['body']} | URL: {r['href']}")
            return "\n\n".join(results) if results else "Nessun dato web trovato."
    except Exception:
        return "Errore connessione ricerca."

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_data: Dict[str, str]
    next_node: str

class TravelAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
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
        prompt = """Analizza la chat ed estrai le info. Se mancano, chiedile. 
        Se hai TUTTO, scrivi 'READY_TO_SEARCH' e riassumi: Partenza, Destinazione, Date, Budget, Interessi, Gruppo."""
        res = self.llm.invoke([HumanMessage(content=prompt)] + state['messages'])
        is_ready = "READY_TO_SEARCH" in res.content
        return {
            "messages": [res],
            "next_node": "research" if is_ready else "ask",
            "user_data": {"summary": res.content}
        }

    # Metodo helper per la Sidebar (Pag. 289: Trasparenza)
    def get_structured_data(self, history: list):
        prompt = """Analizza la conversazione e restituisci SOLO i valori estratti per questi campi:
        Partenza, Destinazione, Date, Budget, Interessi, Gruppo.
        Usa 'Mancante' se il dato non è presente. Rispondi con un elenco puntato."""
        res = self.llm.invoke([HumanMessage(content=prompt)] + history)
        return res.content

    def _research_node(self, state: AgentState):
        summary = state["user_data"]["summary"]
        return {
            "user_data": {
                "flights": travel_web_search(f"voli reali prezzi per {summary}"),
                "hotels": travel_web_search(f"hotel reali prezzi per {summary}"),
                "summary": summary
            }
        }

    def _architect_node(self, state: AgentState):
        data = state["user_data"]
        prompt = f"""Crea un itinerario per la destinazione richiesta: {data['summary']}.
        Usa i dati web: Voli: {data['flights']}, Hotel: {data['hotels']}.
        Rispetta il budget. Se mancano dati reali, usa la tua conoscenza per hotel reali a {data['summary']}."""
        res = self.llm.invoke(prompt)
        return {"messages": [res]}

    def run(self, user_input: str, history: list):
        inputs = {"messages": history + [HumanMessage(content=user_input)], "user_data": {}, "next_node": ""}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content
