import pandas as pd
import pickle
import questionary
from menu import limpar_tela

def carregar_artefatos():
    """Carrega todos os artefatos necessários (modelos, normalizador, colunas)."""
    try:
        with open('./models/modelo_normalizador_num.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('./models/modeloDiagnosis.pkl', 'rb') as f:
            modelDiagnosis = pickle.load(f)
        with open('./models/modeloSeverity.pkl', 'rb') as f:
            modelSeverity = pickle.load(f)
        with open('./models/modeloManagement.pkl', 'rb') as f:
            modelManagement = pickle.load(f)
        
        return scaler, modelDiagnosis, modelSeverity, modelManagement
    
    except FileNotFoundError as e:
        print(f"Erro ao carregar o arquivo: {e}")
        print("Certifique-se de que o script de treinamento foi executado e todos os arquivos .pkl foram gerados.")
        return None, None, None, None

def coletar_dados(colunas_para_preencher, numerical_cols, map_cols):
    """Coleta os dados de entrada do usuário de forma interativa com a biblioteca questionary."""
    dados_inferencia = {}
    limpar_tela()
    print("--- Preencha os dados para a inferência ---")
    print("(Pressione Ctrl+C a qualquer momento para cancelar)")

    # Validador para garantir que a entrada é numérica
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return "Por favor, insira um valor numérico."

    try:
        for col in colunas_para_preencher:
            if col in map_cols:
                # Usa um prompt de confirmação para colunas 'yes'/'no'
                resposta = questionary.confirm(f"{col}?", default=False, qmark="?").ask()
                if resposta is None: return None # Cancelado pelo usuário
                dados_inferencia[col] = 'yes' if resposta else 'no'

            elif col in numerical_cols:
                # Usa um prompt de texto com validação para colunas numéricas
                valor_str = questionary.text(
                    f"{col}:", 
                    validate=is_numeric,
                    qmark=":"
                ).ask()
                if valor_str is None: return None # Cancelado pelo usuário
                
                # Converte para float ou int, mantendo a lógica original
                if '.' in valor_str:
                    dados_inferencia[col] = float(valor_str)
                else:
                    dados_inferencia[col] = int(valor_str)
            else:
                # Prompt de texto para outras colunas categóricas (ex: 'Sex')
                valor = questionary.text(f"{col}:", qmark=":").ask()
                if valor is None: return None # Cancelado pelo usuário
                dados_inferencia[col] = valor.strip()

    except KeyboardInterrupt:
        # Permite sair de forma limpa com Ctrl+C
        print("\n\nColeta de dados cancelada.")
        return None

    return pd.DataFrame([dados_inferencia])


def transform_inferencia(df_infer, scaler, reference_columns, numeric_cols, map_cols):
    """
    Transforma o df_infer no mesmo formato de dados_normalizado.
    (Função original mantida sem alterações)
    """
    for col in map_cols:
        if col in df_infer.columns:
            df_infer[col] = df_infer[col].map({'yes': 1, 'no': 0})

    df_num = df_infer[numeric_cols]
    df_num_norm = pd.DataFrame(
        scaler.transform(df_num),
        columns=numeric_cols,
        index=df_infer.index
    )

    df_cat = df_infer.drop(columns=numeric_cols)
    df_cat_dummies = pd.get_dummies(df_cat, prefix_sep='_', dtype=int)

    df_trans = pd.concat([df_num_norm, df_cat_dummies], axis=1)
    df_trans = df_trans.reindex(columns=reference_columns, fill_value=0)

    return df_trans

def inferencia_main():
    """Função principal para executar o processo de inferência."""
    scaler, modelDiagnosis, modelSeverity, modelManagement = carregar_artefatos()
    
    if scaler is None:
        return

    numerical_cols = [
        'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Appendix_Diameter',
        'Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'RBC_Count',
        'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP'
    ]
    
    map_columns = {
        'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
        'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Neutrophilia', 'Dysuria',
        'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids'
    }
    
    df_original = pd.read_csv('./data/df_original.csv')
    
    # Passa as listas de colunas para a função de coleta
    df_infer = coletar_dados(df_original.columns.tolist(), numerical_cols, map_columns)
    
    # Se o usuário cancelou a coleta, df_infer será None
    if df_infer is None or df_infer.empty:
        print("Processo de inferência não continuou.")
        return
    
    df_norm = pd.read_csv('./data/df_normalizado.csv')

    df_trans = transform_inferencia(
        df_infer, 
        scaler, 
        df_norm.columns[:-3].tolist(),
        numerical_cols,
        map_columns
    )

    diagnosis_classes = modelDiagnosis.classes_
    severity_classes = modelSeverity.classes_
    management_classes = modelManagement.classes_

    diag_proba = modelDiagnosis.predict_proba(df_trans)[0]
    diag_pred = diagnosis_classes[diag_proba.argmax()]
    diag_acc = diag_proba.max()
    
    print("\n--- Resultados da Inferência ---")
    print(f"Diagnóstico: {diag_pred} (Acurácia: {diag_acc:.2%})")

    if diag_pred == 'appendicitis':
        sev_proba = modelSeverity.predict_proba(df_trans)[0]
        sev_pred = severity_classes[sev_proba.argmax()]
        sev_acc = sev_proba.max()
        
        mgmt_proba = modelManagement.predict_proba(df_trans)[0]
        mgmt_pred = management_classes[mgmt_proba.argmax()]
        mgmt_acc = mgmt_proba.max()
        
        print(f"Severidade: {sev_pred} (Acurácia: {sev_acc:.2%})")
        print(f"Cuidados: {mgmt_pred} (Acurácia: {mgmt_acc:.2%})")