import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
DEFAULT_COLLECTION_NAME = "document_chunks"

TOP_K = 10

UNKNOWN_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

def _detect_provider() -> str:
    configured = os.getenv("LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured

    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"

    if (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    ):
        return "google"

    return "openai"


LLM_PROVIDER = _detect_provider()

DATABASE_URL = (
    os.getenv("DATABASE_URL", "").strip()
    or DEFAULT_DATABASE_URL
)
COLLECTION_NAME = (
    os.getenv("PG_VECTOR_COLLECTION_NAME", "").strip()
    or DEFAULT_COLLECTION_NAME
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
).strip()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini").strip()

GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2"
).strip()
GOOGLE_LLM_MODEL = os.getenv(
    "GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite"
).strip()


def _resolve_pdf_path() -> Path:
    configured_path = os.getenv("PDF_PATH", "document.pdf").strip() or "document.pdf"
    path = Path(configured_path).expanduser()

    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return BASE_DIR / path


PDF_PATH = _resolve_pdf_path()


def _provider() -> str:
    if LLM_PROVIDER in {"google", "gemini"}:
        return "google"
    if LLM_PROVIDER == "openai":
        return "openai"

    raise RuntimeError(
        "LLM_PROVIDER inválido. Use 'openai' ou 'google' no arquivo .env."
    )


def _google_api_key() -> str:
    return (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )


def get_embeddings():
    provider = _provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. Preencha o arquivo .env."
            )

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = _google_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY não configurada. Preencha o arquivo .env."
        )

    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def get_llm():
    provider = _provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. Preencha o arquivo .env."
            )

        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            api_key=api_key,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = _google_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY não configurada. Preencha o arquivo .env."
        )

    return ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        api_key=api_key,
        temperature=0,
    )


def get_vector_store():
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
