from menu import exibir_menu, esperar_enter, limpar_tela
from inferencia import inferir
from treinadora import treinar

limpar_tela()

itens = ['Treinamento', 'Inferência']

op = int(-1)

while op != 0:
    
    op = int(exibir_menu("Pediatric Appendicitis", itens, True))
    
    if op == 1:
        limpar_tela()
        treinar()
        esperar_enter()
    elif op == 2:
        limpar_tela()
        inferir()
        esperar_enter()
    elif op == 0:
        break
    else:
        limpar_tela()
        print("Opção inválida. Tente novamente.")
        esperar_enter()