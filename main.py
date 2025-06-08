from menu import exibir_menu, esperar_enter, limpar_tela
from inferencia import inferencia_main
from treinadora import treinar

def main():
    limpar_tela()
    itens = ['Treinamento', 'Inferência']
    op = -1

    while op != 0:
        try:
            op = int(exibir_menu("Pediatric Appendicitis", itens, True))
            
            if op == 1:
                limpar_tela()
                treinar()
                esperar_enter()
            elif op == 2:
                limpar_tela()
                inferencia_main()
                esperar_enter()
            elif op == 0:
                break
            else:
                limpar_tela()
                print("Opção inválida. Tente novamente.")
                esperar_enter()
        except ValueError:
            limpar_tela()
            print("Entrada inválida. Digite um número.")
            esperar_enter()

if __name__ == "__main__":
    main()