<img src="./img/apendicite.png" width="150px">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Pediatric Appendicitis Predictor

## Descrição

### O que é?

O **Pediatric Appendicitis Predictor** é uma ferramenta de linha de comando desenvolvida em Python para auxiliar na predição de apendicite em pacientes pediátricos. Utilizando um modelo de Machine Learning treinado com o conjunto de dados [Regensburg Pediatric Appendicitis](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis), a aplicação é capaz de prever:

1.  **Diagnóstico**: Se o paciente tem ou não apendicite.
2.  **Severidade**: Caso o diagnóstico seja positivo, qual a gravidade da condição.
3.  **Manejo Clínico**: Qual a recomendação de tratamento (cirúrgico ou não cirúrgico).

A aplicação possui duas funcionalidades principais: **Inferência**, para fazer previsões com base em novos dados de pacientes, e **Treinamento**, que permite regenerar os modelos de Machine Learning a partir do zero.

### Como funciona?

A aplicação opera em dois modos distintos que podem ser selecionados através de um menu interativo:

#### 1. Modo de Inferência

Neste modo, o usuário fornece interativamente os dados clínicos de um paciente. O sistema faz uma série de perguntas, uma para cada característica necessária (ex: Idade, Temperatura Corporal, Dor Migratória, etc.).

-   Os dados fornecidos são pré-processados e normalizados da mesma forma que os dados de treinamento.
-   O primeiro modelo (Diagnóstico) prevê se há apendicite.
-   Se o diagnóstico for positivo para apendicite, outros dois modelos são executados em sequência para prever a Severidade e o Manejo clínico recomendado.
-   Ao final, os resultados da predição, junto com a confiança (acurácia) de cada modelo, são exibidos na tela.

#### 2. Modo de Treinamento

Esta funcionalidade permite que o usuário treine novamente os modelos de Machine Learning. O processo automatizado realiza as seguintes etapas:

-   **Busca e Preparação dos Dados**: O conjunto de dados é baixado do repositório da UCI.
-   **Pré-processamento**: Os dados são limpos, valores ausentes são tratados (imputação por mediana para dados numéricos e por moda para categóricos), e as variáveis são normalizadas (`MinMaxScaler`) e transformadas (`One-Hot Encoding`).
-   **Balanceamento**: Para corrigir o desequilíbrio entre as classes (ex: mais casos negativos do que positivos), a técnica **SMOTE** (Synthetic Minority Over-sampling Technique) é aplicada.
-   **Treinamento de Modelos**: Três modelos `RandomForestClassifier` são treinados separadamente para Diagnóstico, Severidade e Manejo. O processo utiliza `RandomizedSearchCV` para encontrar os melhores hiperparâmetros e validação cruzada para avaliar a performance.
-   **Salvamento dos Artefatos**: Os modelos treinados e o normalizador de dados são salvos como arquivos `.pkl` na pasta `./models/`, para serem usados posteriormente pelo modo de Inferência.

## Pré-requisitos

Para executar esta aplicação, você precisará ter o Python 3 instalado em seu sistema. As seguintes bibliotecas Python são necessárias:

* `questionary`
* `scikit-learn`
* `imbalanced-learn`
* `ucimlrepo`
* `pandas`

## Instruções de instalação

Siga os passos abaixo para configurar e executar a aplicação.

1.  **Clone o repositório (ou crie a pasta do projeto)**
    Se seu projeto estiver em um repositório git, clone-o. Caso contrário, apenas certifique-se de que todos os arquivos (`main.py`, `treinadora.py`, `inferencia.py`, `menu.py`) estão na mesma pasta.

2.  **Crie e ative um ambiente virtual**
    É uma boa prática usar um ambiente virtual para isolar as dependências do projeto. Abra seu terminal ou prompt de comando na pasta do projeto e execute:

    ```bash
    # Cria o ambiente virtual (substitua 'venv' pelo nome que preferir)
    python -m venv venv

    # Ativa o ambiente virtual
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as dependências**
    Com o ambiente virtual ativado, instale todas as bibliotecas necessárias com um único comando:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Crie as pastas necessárias**
    A aplicação precisa de pastas específicas para salvar os modelos e os dados gerados. Crie-as com os seguintes comandos:

    ```bash
    # No Windows
    mkdir models
    mkdir data

    # No macOS/Linux
    mkdir -p models data
    ```

5.  **Execute o script de Treinamento**
    Antes de usar a inferência pela primeira vez, você precisa treinar os modelos. Execute o arquivo principal e escolha a opção "Treinamento".

    ```bash
    python main.py
    ```
    Selecione `Treinamento` no menu e confirme. Este processo pode levar alguns minutos.

6.  **Execute a Inferência**
    Após o treinamento ser concluído com sucesso, você pode executar a aplicação novamente para fazer previsões.

    ```bash
    python main.py
    ```
    Escolha `Inferência` no menu e siga as instruções para inserir os dados do paciente.