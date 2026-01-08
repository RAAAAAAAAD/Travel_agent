import pytest
from app.agent_logic import TravelAgent

def test_agent_initialization():
    # Verifica che il grafo sia compilato correttamente
    agent = TravelAgent()
    assert agent.app is not None

def test_custom_search():
    # Verifica che l'import di DDGS funzioni ora
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        assert ddgs is not None
