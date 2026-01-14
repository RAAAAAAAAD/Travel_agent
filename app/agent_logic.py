import os
from typing import TypedDict, Annotated, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Cerca il .env salendo di una cartella
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class TravelProfile(BaseModel):
    partenza: str = Field(default="Mancante")
    destinazione: str = Field(default="Mancante")
    date: str = Field(default="Mancante")
    budget: str = Field(default="Mancante")
    interessi: str = Field(default="Mancante")
    gruppo: str = Field(default="Mancante")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    profile: TravelProfile
    next_node: str

class TravelAgent:
    def __init__(self):
        # .strip() rimuove spazi accidentali
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY non trovata. Controlla il file .env alla radice!")
        
        self.llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=groq_key)
        self.search = TavilySearchResults(max_results=5, search_depth="advanced", tavily_api_key=tavily_key)

        workflow = StateGraph(AgentState)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("architect", self._architect_node)
        
        workflow.set_entry_point("analyzer")
        workflow.add_conditional_edges("analyzer", self._router, {"research": "researcher", "ask": END})
        workflow.add_edge("researcher", "architect")
        workflow.add_edge("architect", END)
        self.app = workflow.compile()

    def _analyzer_node(self, state: AgentState):
        structured_llm = self.llm.with_structured_output(TravelProfile)
        new_profile = structured_llm.invoke(state["messages"])
        
        # Campi minimi per procedere (Planning Pattern)
        fields = ["partenza", "destinazione", "date", "budget", "interessi", "gruppo"]
        ready = all(getattr(new_profile, f) != "Mancante" for f in fields)
        
        prompt = f"Sei un Travel Agent. Dati correnti: {new_profile.json()}. Se manca qualcosa chiedila, altrimenti conferma l'inizio della ricerca."
        res = self.llm.invoke([HumanMessage(content=prompt)] + state["messages"])
        
        return {"messages": [res], "profile": new_profile, "next_node": "research" if ready else "ask"}

    def _router(self, state: AgentState):
        return state["next_node"]

    def _research_node(self, state: AgentState):
        p = state["profile"]
        q_flights = f"prezzi voli aerei e compagnie da {p.partenza} a {p.destinazione} {p.date}"
        q_hotels = f"nomi hotel e prezzi a {p.destinazione} budget {p.budget} booking.com"
        
        res_flights = self.search.invoke(q_flights)
        res_hotels = self.search.invoke(q_hotels)
        return {"messages": [AIMessage(content=f"VOLI: {res_flights}\n\nHOTEL: {res_hotels}")]}

    def _architect_node(self, state: AgentState):
        raw_data = state["messages"][-1].content
        prompt = f"""Crea un itinerario per {state['profile'].destinazione}.
        DATI REALI: {raw_data}
        DEVI INCLUDERE TABELLE CON NOMI, PREZZI E LINK CLICCABILI."""
        res = self.llm.invoke(prompt)
        return {"messages": [res]}

    def run(self, user_input: str, history: list, profile: TravelProfile):
        inputs = {"messages": history + [HumanMessage(content=user_input)], "profile": profile, "next_node": ""}
        return self.app.invoke(inputs)
