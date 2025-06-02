from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from pickle import dump
import pandas as pd


def treinar():
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)
    X = regensburg_pediatric_appendicitis.data.features
    y = regensburg_pediatric_appendicitis.data.targets
    df = pd.concat([X, y], axis=1)
    
    # salvar o DataFrame normalizado
    df.to_csv('df_normalizado.csv', index=False)

    df = normalizar_targets(df)

    df = normalizar(df)

    print("\nTreinando modelo para Diagnosis...")
    df_diagnosis = df.copy()
    df_diagnosis = balancear(df_diagnosis, 'Diagnosis')
    treinar_modelo(df_diagnosis, 'Diagnosis')

    print("\nTreinando modelo para Severity...")
    df_severity = df[df['Diagnosis'] == 'appendicitis'].copy()
    df_severity = balancear(df_severity, 'Severity')
    treinar_modelo(df_severity, 'Severity')

    print("\nTreinando modelo para Management...")
    df_management = df[df['Diagnosis'] == 'appendicitis'].copy()
    df_management = balancear(df_management, 'Management')
    treinar_modelo(df_management, 'Management')


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os atributos (features) via MinMaxScaler e aplica One-Hot-Encoding nas colunas categóricas.
    Não tenta converter as colunas de target em valores numéricos: elas são separadas, processadas as features
    e depois reanexadas.
    """

    target_cols = ['Diagnosis', 'Severity', 'Management']
    present_targets = [col for col in target_cols if col in df.columns]
    df_targets = df[present_targets]
    df_features = df.drop(columns=present_targets)

    initial_columns = df_features.shape[1]
    df_features = df_features.loc[:, df_features.isnull().mean() < 0.5]

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
    for col in map_columns:
        if col in df_features.columns:
            df_features[col] = df_features[col].map({'yes': 1, 'no': 0})

    categorical_cols_user = [
        'Sex',
        'Ketones_in_Urine',
        'RBC_in_Urine',
        'WBC_in_Urine',
        'Stool',
        'Peritonitis',
    ]
    categorical_cols = [col for col in categorical_cols_user if col in df_features.columns]
    for col in categorical_cols:
        if df_features[col].isnull().any():
            try:
                mode_val = df_features[col].mode()[0]
                df_features[col] = df_features[col].fillna(mode_val)
            except Exception as e:
                print(
                    f"Erro ao imputar moda para coluna '{col}': {e}. "
                    f"Dtype: {df_features[col].dtype}"
                )

    candidate_cols = [col for col in df_features.columns if col not in categorical_cols]
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            if df_features[col].isnull().any():
                median_val = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_val)
        else:
            try:
                df_features[col] = pd.to_numeric(df_features[col])
                if df_features[col].isnull().any():
                    median_val = df_features[col].median()
                    df_features[col] = df_features[col].fillna(median_val)
            except ValueError:
                print(
                    f"Coluna '{col}' foi identificada como numérica mas seu dtype é "
                    f"{df_features[col].dtype} e não pôde ser convertida. "
                    "Nenhuma imputação de mediana feita para esta coluna."
                )

    df_features = df_features.dropna()
    df_targets = df_targets.loc[df_features.index]

    numerical_cols = [col for col in df_features.columns if pd.api.types.is_numeric_dtype(df_features[col])]
    normalizador = preprocessing.MinMaxScaler()
    modelo_normalizador_num = normalizador.fit(df_features[numerical_cols])
    dump(modelo_normalizador_num, open('./models/modelo_normalizador_num.pkl', 'wb'))
    df_features[numerical_cols] = modelo_normalizador_num.transform(df_features[numerical_cols])

    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)

    df_processed = pd.concat(
        [df_features.reset_index(drop=True), df_targets.reset_index(drop=True)],
        axis=1
    )
    return df_processed


def normalizar_targets(df_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes nas colunas de target: 'Diagnosis', 'Management' e 'Severity'.
    """
    categorical_cols_user = [
        'Diagnosis',
        'Management',
        'Severity'
    ]
    present_cols = [col for col in categorical_cols_user if col in df_targets.columns]
    for col in present_cols:
        if df_targets[col].isnull().any():
            try:
                mode_val = df_targets[col].mode()[0]
                df_targets[col] = df_targets[col].fillna(mode_val)
            except Exception as e:
                print(
                    f"Erro ao imputar moda para coluna '{col}': {e}. "
                    f"Dtype: {df_targets[col].dtype}"
                )
    return df_targets


def balancear(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Faz oversampling com SMOTE na coluna de classes 'column'.
    Assume que o DataFrame já contenha apenas aquela coluna de destino e as features.
    """
    df_classes = df[column]
    df_atributos = df.drop(columns=['Diagnosis', 'Management', 'Severity'], errors='ignore')
    resampler = SMOTE()
    df_atributos_b, df_classes_b = resampler.fit_resample(df_atributos, df_classes)
    df_balanced = pd.concat([df_atributos_b, df_classes_b], axis=1)
    return df_balanced


def treinar_modelo(df: pd.DataFrame, column: str):
    """
    Treina um RandomForestClassifier para a coluna 'column', rodando
    RandomizedSearchCV + validação cruzada. Salva o modelo em 'modelo<column>.pkl'.
    """
    df_atributos = df.drop(columns=[column])
    df_classes = df[column]

    atr_train, atr_test, cls_train, cls_test = train_test_split(
        df_atributos, df_classes, test_size=0.3, random_state=42
    )

    n_estimators = [100, 200, 300]
    max_depth = [5, 10, 15, 20]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_features = ['sqrt', 'log2']
    bootstrap = [True, False]
    criterion = ['gini', 'entropy', 'log_loss']

    forest_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'criterion': criterion
    }

    forest_hyperparameters = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=forest_grid,
        n_iter=10,
        cv=5,
        verbose=1,
        random_state=42
    )

    forest_hyperparameters.fit(atr_train, cls_train)

    forest_final = RandomForestClassifier(**forest_hyperparameters.best_params_)
    forest_model = forest_final.fit(df_atributos, df_classes)

    scoring = ['precision_macro', 'recall_macro']
    scores_cross = cross_validate(forest_final, df_atributos, df_classes, cv=10, scoring=scoring)

    print(f"Scores de validação cruzada (Target: {column}) :")
    print("  Precisão média:", scores_cross['test_precision_macro'].mean())
    print("  Recall médio:   ", scores_cross['test_recall_macro'].mean())

    salvar_modelo(column, forest_model)


def salvar_modelo(column: str, modelo):
    """
    Salva o modelo em 'modelo<column>.pkl'.
    """
    dump(modelo, open(f'./models/modelo{column}.pkl', 'wb'))
