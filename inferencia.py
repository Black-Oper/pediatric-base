import pandas as pd
from pickle import load
from ucimlrepo import fetch_ucirepo

from treinadora import normalizar

def carregar_dados_crus() -> pd.DataFrame:
    """
    Busca a base de dados (UCI Pediatric Appendicitis, id=938),
    concatena features e targets e retorna o DataFrame bruto.
    """
    rep = fetch_ucirepo(id=938)
    X = rep.data.features
    y = rep.data.targets
    df = pd.concat([X, y], axis=1)
    return df

def obter_input_usuario(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pergunta ao usuário, via input(), o valor de cada coluna de feature (atributo).
    Preenche as colunas de target com pd.NA para que depois o normalizar()
    funcione sem problemas. Retorna um DataFrame com uma única linha.
    """

    target_cols = ['Diagnosis', 'Severity', 'Management']
    feature_cols = [c for c in df_raw.columns if c not in target_cols]

    print(">>> Insira os valores para a NOVA instância:")
    dados = {}
    for col in feature_cols:
        val = input(f"  {col}: ")
        dados[col] = val

    for tgt in target_cols:
        if tgt in df_raw.columns:
            dados[tgt] = pd.NA

    df_novo = pd.DataFrame([dados])
    return df_novo

def inferir():
    df_raw = carregar_dados_crus()

    df_novo = obter_input_usuario(df_raw)

    df_combined = pd.concat([df_raw, df_novo], ignore_index=True)

    df_processed = normalizar(df_combined)

    target_cols = ['Diagnosis', 'Severity', 'Management']
    feature_cols_processed = [c for c in df_processed.columns if c not in target_cols]

    X_novo = df_processed.iloc[[-1]][feature_cols_processed]

    model_diag = load(open('modeloDiagnosis.pkl', 'rb'))
    model_sev  = load(open('modeloSeverity.pkl',  'rb'))
    model_mgmt = load(open('modeloManagement.pkl', 'rb'))

    pred_diag = model_diag.predict(X_novo)[0]
    print(f"\n>>> Diagnosis previsto: {pred_diag!r}")

    if pred_diag == 'appendicitis':
        pred_sev = model_sev.predict(X_novo)[0]
        print(f">> Severity previsto: {pred_sev!r}")

        pred_mgmt = model_mgmt.predict(X_novo)[0]
        print(f">> Management previsto: {pred_mgmt!r}")
    else:
        print(">> Sem apendicite: Severity e Management não se aplicam.\n")
