import pandas as pd
import pickle
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

def coletar_dados(colunas_para_preencher):
    """Coleta os dados de entrada do usuário."""
    dados_inferencia = {}
    limpar_tela()
    print("--- Preencha os dados para a inferência ---")

    for col in colunas_para_preencher:

        idx = colunas_para_preencher.index(col)
        valor = input(f"{idx + 1}. {col}: ").strip()
        
        if valor is not None and valor != '':
            try:

                if isinstance(valor, str) and '.' in valor:
                    dados_inferencia[col] = float(valor)

                elif isinstance(valor, (int, float)) or (isinstance(valor, str) and valor.isdigit()):
                    dados_inferencia[col] = int(valor)
                else:
                    dados_inferencia[col] = valor
            except (ValueError, TypeError):
                dados_inferencia[col] = valor
        else:
            print(f"Valor para {col} não pode ser vazio. Tente novamente.")
                
    return pd.DataFrame([dados_inferencia])

def transform_inferencia(df_infer, scaler, reference_columns, numeric_cols, map_cols):
    """
    Transforma o df_infer no mesmo formato de dados_normalizado.

    Parâmetros
    ----------
    df_infer : pd.DataFrame
        DataFrame bruto de inferência, com colunas originais (numéricas + categóricas).
    scaler : objeto sklearn transformer
        Scaler já ajustado em dados_num (por exemplo, StandardScaler ou MinMaxScaler).
    reference_columns : list of str
        Lista exata de colunas (ordem incluída) de dados_normalizado.
    numeric_cols : list of str
        Nome das colunas numéricas que foram normalizadas em dados_normalizado.

    Retorna
    -------
    pd.DataFrame
        DataFrame transformado, com colunas = reference_columns.
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

    df_original = pd.read_csv('df_original.csv')
    
    df_infer = coletar_dados(df_original.columns.tolist())
    if df_infer.empty:
        return
    
    df_norm = pd.read_csv('df_normalizado.csv')
    
    numerical_cols = [
        'Age',
        'BMI',
        'Height',
        'Weight',
        'Length_of_Stay',
        'Appendix_Diameter',
        'Body_Temperature',
        'WBC_Count',
        'Neutrophil_Percentage',
        'RBC_Count',
        'Hemoglobin',
        'RDW',
        'Thrombocyte_Count',
        'CRP'
    ]
    
    map_columns = {
        'Appendix_on_US',
        'Migratory_Pain',
        'Lower_Right_Abd_Pain',
        'Contralateral_Rebound_Tenderness',
        'Coughing_Pain',
        'Nausea',
        'Loss_of_Appetite',
        'Neutrophilia',
        'Dysuria',
        'Psoas_Sign',
        'Ipsilateral_Rebound_Tenderness',
        'US_Performed',
        'Free_Fluids'
    }

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