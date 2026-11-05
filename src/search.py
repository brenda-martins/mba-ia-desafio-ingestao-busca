import sys
from typing import Callable, Optional, Union

from langchain_core.messages import HumanMessage

from config import TOP_K, UNKNOWN_ANSWER, get_llm, get_vector_store

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(part for part in parts if part)

    return str(content)


def build_chain() -> Callable[[str], str]:
    vector_store = get_vector_store()
    llm = get_llm()

    def ask(question: str) -> str:
        question = question.strip()
        if not question:
            return UNKNOWN_ANSWER

        results = vector_store.similarity_search_with_score(question, k=TOP_K)
        if not results:
            return UNKNOWN_ANSWER

        context_parts = []
        for document, _score in results:
            content = document.page_content.strip()
            if content:
                context_parts.append(content)

        if not context_parts:
            return UNKNOWN_ANSWER

        prompt = PROMPT_TEMPLATE.format(
            contexto="\n\n---\n\n".join(context_parts),
            pergunta=question,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        answer = _extract_text(response.content).strip()

        return answer or UNKNOWN_ANSWER

    return ask


def search_prompt(
    question: Optional[str] = None,
) -> Optional[Union[str, Callable[[str], str]]]:
  
    try:
        ask = build_chain()
    except Exception as exc:
        print(f"Erro ao inicializar a busca: {exc}", file=sys.stderr)
        return None

    if question is not None:
        return ask(question)

    return ask
