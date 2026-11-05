import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    COLLECTION_NAME,
    DATABASE_URL,
    PDF_PATH,
    get_embeddings,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF não encontrado em '{PDF_PATH}'. "
            "Ajuste a variável PDF_PATH no arquivo .env."
        )

    print(f"Carregando PDF: {PDF_PATH}")
    documents = PyPDFLoader(str(PDF_PATH)).load()

    if not documents:
        raise RuntimeError("Nenhum conteúdo foi extraído do PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError("O PDF não gerou nenhum chunk para ingestão.")

    print(f"Páginas carregadas: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Banco: {DATABASE_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print("Gerando embeddings e gravando no PostgreSQL/pgVector...")

    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        print(f"Erro na ingestão: {exc}", file=sys.stderr)
        sys.exit(1)
