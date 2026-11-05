# Ingestão e Busca Semântica com LangChain e PostgreSQL/pgVector

Sistema RAG (Retrieval-Augmented Generation) que lê um PDF, salva seus trechos como
embeddings no PostgreSQL com pgVector e responde perguntas pelo terminal usando
somente o conteúdo do documento.

## Como funciona

1. `src/ingest.py` carrega o PDF com `PyPDFLoader`.
2. O texto é dividido em chunks de **1000 caracteres** com **overlap de 150**.
3. Cada chunk é convertido em embedding pelo provedor escolhido (OpenAI ou Gemini).
4. Os vetores são gravados na collection configurada no PostgreSQL/pgVector.
5. `src/chat.py` lê a pergunta, vetoriza, busca os **10 trechos mais relevantes**
   (`k=10`), monta o prompt com o contexto e chama a LLM.
6. A resposta é exibida no terminal. Se a resposta não estiver no contexto, a LLM
   deve responder: `Não tenho informações necessárias para responder sua pergunta.`

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py         # Configuração e criação dos modelos/vector store
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca semântica + prompt + chamada da LLM
│   └── chat.py           # CLI de conversação
├── document.pdf
└── README.md
```

## Pré-requisitos

- Python **3.10+** (recomendado 3.11 ou superior)
- Docker e Docker Compose
- Uma API Key da OpenAI **ou** do Google Gemini

## Configuração

### 1. Crie e ative o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

No Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e escolha o provedor com `LLM_PROVIDER=openai` ou
`LLM_PROVIDER=google`. Se `LLM_PROVIDER` não for informado, o sistema detecta o
provedor pela chave disponível (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`).

#### Opção A — OpenAI

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sua-chave
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=document_chunks
PDF_PATH=document.pdf
```

#### Opção B — Google Gemini

```dotenv
LLM_PROVIDER=google
GOOGLE_API_KEY=sua-chave
GOOGLE_EMBEDDING_MODEL=gemini-embedding-2
GOOGLE_LLM_MODEL=gemini-3.5-flash-lite
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=document_chunks
PDF_PATH=document.pdf
```

> Os nomes dos modelos mudam com frequência. Antes de executar, confira os modelos
> disponíveis na documentação oficial do provedor escolhido:
>
> - OpenAI: <https://platform.openai.com/docs/models>
> - Gemini: <https://ai.google.dev/gemini-api/docs/models>
>
> Para este desafio, modelos leves/baratos são suficientes. Basta alterar as
> variáveis `*_MODEL` no `.env`, sem mudar o código.

### Variáveis de ambiente

| Variável | Descrição | Padrão |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` ou `google` | Detectado pela chave; fallback `openai` |
| `OPENAI_API_KEY` | API Key da OpenAI | — |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings da OpenAI | `text-embedding-3-small` |
| `OPENAI_LLM_MODEL` | Modelo de chat da OpenAI | `gpt-4o-mini` |
| `GOOGLE_API_KEY` | API Key do Google Gemini | — |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings do Gemini | `gemini-embedding-2` |
| `GOOGLE_LLM_MODEL` | Modelo de chat do Gemini | `gemini-3.5-flash-lite` |
| `DATABASE_URL` | URL do PostgreSQL com driver `psycopg` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `document_chunks` |
| `PDF_PATH` | Caminho do PDF | `document.pdf` |

## Execução

### 1. Suba o banco de dados

```bash
docker compose up -d
```

Se a sua instalação usa o Compose standalone, use:

```bash
docker-compose up -d
```

O `docker-compose.yml` sobe o PostgreSQL com a imagem `pgvector/pgvector:pg17` e
executa o `CREATE EXTENSION IF NOT EXISTS vector;` automaticamente.

Confira se o container está saudável:

```bash
docker compose ps
```

### 2. Execute a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada (os números podem variar):

```text
Carregando PDF: /caminho/do/projeto/document.pdf
Páginas carregadas: 34
Chunks gerados: 67
Banco: postgresql+psycopg://postgres:postgres@localhost:5432/rag
Collection: document_chunks
Gerando embeddings e gravando no PostgreSQL/pgVector...
Ingestão concluída com sucesso.
```

Executar a ingestão novamente recria a collection, evitando chunks duplicados.

### 3. Inicie o chat

```bash
python src/chat.py
```

Exemplo de uso:

```text
Faça sua pergunta (digite 'sair' para encerrar).

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da Empresa SuperTechIABrazil é de R$ 10.000.000,00.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair`, `exit` ou `quit` para encerrar. `Ctrl+C` também encerra o chat.

## Detalhes da implementação

- **Chunking:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`.
- **Armazenamento:** `PGVector.from_documents(...)` com `use_jsonb=True`.
- **Busca:** `vector_store.similarity_search_with_score(pergunta, k=10)`.
- **Prompt:** o prompt do desafio está em `src/search.py` e instrui a LLM a responder
  somente com base no contexto, sem usar conhecimento externo.
- **Provedores:** o código suporta OpenAI e Gemini por meio da variável
  `LLM_PROVIDER`; as classes de embeddings e LLM são criadas em `src/config.py`.

## Solução de problemas

### `connection refused` ao executar a ingestão

O banco ainda não subiu. Execute `docker compose up -d` e aguarde o healthcheck:

```bash
docker compose ps
docker compose logs postgres
```

### Extensão `vector` não encontrada

O serviço `bootstrap_vector_ext` do Compose cria a extensão automaticamente. Se
necessário, rode manualmente:

```bash
docker compose exec postgres psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Erro de dimensão dos vetores

Modelos de embeddings diferentes geram vetores de dimensões diferentes. A
ingestão recria a collection configurada antes de gravar os novos vetores, então
normalmente basta executar `python src/ingest.py` novamente ao trocar de modelo.
Se ainda ocorrer erro de dimensão (por exemplo, ao usar uma collection criada
manualmente), apague o volume e refaça a ingestão:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

> `docker compose down -v` remove o volume do PostgreSQL e, portanto, todos os
> dados indexados. Use apenas quando quiser recriar a base do zero.

### Erro de API Key ou modelo não encontrado

- Confirme que a chave correta (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`) está no `.env`.
- Confirme que `LLM_PROVIDER` corresponde ao provedor da chave.
- Verifique se o modelo configurado ainda está disponível na documentação oficial.

### `No module named ...` ao executar os scripts

Ative o ambiente virtual antes de instalar e executar:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Requisitos atendidos

- [x] Ingestão de PDF com LangChain e `PyPDFLoader`.
- [x] Chunks de 1000 caracteres com overlap de 150.
- [x] Embeddings persistidos em PostgreSQL com pgVector.
- [x] Busca dos 10 resultados mais relevantes (`k=10`).
- [x] Prompt restritivo ao contexto do PDF.
- [x] CLI de perguntas e respostas.
- [x] Banco executado via Docker Compose.
