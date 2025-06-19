from unittest.mock import MagicMock, patch

import pytest

from chatbot.context_message_with_memory import Bimo


@pytest.mark.parametrize("mock_content, expected", [
    ("fora do escopo", "não posso falar sobre isso"),
    ("saudação",       "olá"),
    ("qualquer outro", None),
])
def test_check_constraint_parametrizado(mock_content, expected):
    # GIVEN: um Bimo com mocks para llm, prompts e retriever
    mock_llm = MagicMock()
    mock_system = MagicMock()
    mock_check_context = MagicMock()
    mock_contextualizer = MagicMock()
    mock_constraint_prompt = MagicMock()
    mock_memory = MagicMock()
    mock_retriever = MagicMock()

    bimo = Bimo(
        llm                   = mock_llm,
        system_message        = mock_system,
        check_context_prompt  = mock_check_context,
        contextualizer_prompt = mock_contextualizer,
        constraint_prompt     = mock_constraint_prompt,
        memory                = mock_memory,
        retriever             = mock_retriever,
        include_memory        = False,
        max_messages          = 5
    )

    
    # Cria o mock_chain e o mock_response que queremos “injetar”
    mock_chain    = MagicMock()
    mock_response = MagicMock()
    mock_response.content = mock_content
    mock_chain.invoke.return_value = mock_response

    # 🔧 Aqui vem o truque:
    # 1) dict | prompt  → prompt.__ror__(dict) deve devolver mock_chain
    mock_constraint_prompt.__ror__.return_value = mock_chain
    # 2) mock_chain | llm → mock_chain.__or__(dummy_llm) deve devolver mock_chain de novo
    mock_chain.__or__.return_value      = mock_chain

    # WHEN
    result = bimo.check_constraint("qualquer coisa")

    # THEN
    assert result == expected