import questionary
from questionary import Separator
from menu import esperar_enter, limpar_tela
from inferencia import inferencia_main
from treinadora import treinar

def main():
    """Função principal que gerencia o menu da aplicação."""
    limpar_tela()

    while True:
        limpar_tela()
        choice = questionary.select(
            "Pediatric Appendicitis :: O que você deseja fazer?",
            choices=[
                Separator(),
                'Inferência',
                'Treinamento',
                'Sair'
            ],
            qmark=">",
            pointer="👉"
        ).ask()

        if choice == 'Inferência':
            limpar_tela()
            inferencia_main()
            esperar_enter()
        elif choice == 'Treinamento':
            limpar_tela()
            treinar()
            esperar_enter()
        elif choice == 'Sair' or choice is None:
            break

    print("Programa finalizado.")

if __name__ == "__main__":
    main()