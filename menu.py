def exibir_menu(titulo, itens, sair):
    limpar_tela()
    
    completo = "Python :: " + titulo
    print(completo)
    print("=" * len(completo))
    
    exibir_itens(itens)
    
    if sair:
        print("[ 0 ] - Sair")
    else:
        print("[ 0 ] - Voltar")
        
    print("=" * len(completo))
    
    linha = input("Escolha uma opção: ")
    
    return linha

def exibir_itens(itens):
    for idx, item in enumerate(itens, start=1):
        print(f"[ {idx} ] - {item}")
        
def esperar_enter():
    input("Pressione Enter para continuar...")
    
    
def limpar_tela():
    print("\033c", end="")