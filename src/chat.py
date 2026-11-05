from search import search_prompt


def main() -> None:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar).")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            break

        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        try:
            answer = chain(question)
        except Exception as exc:
            print(f"RESPOSTA: Erro ao consultar o banco vetorial ou a LLM: {exc}")
            continue

        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()
