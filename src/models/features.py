"""Variáveis de entrada comuns ao HDBSCAN* aprimorado e aos modelos supervisionados."""

import numpy as np

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def engineer_features(X):
    """
    Constrói as variáveis de entrada dos modelos a partir das colunas originais da base.

    Objetivo:
        Mantém os componentes principais V1 a V28 e substitui Amount e Time por três variáveis
        derivadas. O valor da transação entra como log_amount = ln(1 + Amount), transformação
        que comprime a cauda direita da distribuição, fortemente assimétrica em dados
        financeiros, e permanece definida quando Amount é zero. O tempo é reduzido à posição da
        transação no ciclo de 24 horas, hora = (Time mod 86400) / 3600, e codificado como o par
        (sen(2π·hora/24), cos(2π·hora/24)). A codificação cíclica preserva a proximidade entre
        23h e 0h, que uma escala linear trataria como extremos opostos, e o par seno e cosseno
        identifica cada hora de forma única. Como Time conta os segundos desde a primeira
        transação da base, a hora obtida é relativa a esse instante, e não ao horário do relógio.

        O valor bruto de Time não entra: ao longo dos dois dias da base, ele só ordena as
        transações, como um identificador. O HDBSCAN* aprimorado e os modelos supervisionados
        usam esta mesma função, de modo que todos recebem as mesmas variáveis.

    Parâmetros:
        X (pd.DataFrame): transações com as colunas Time, V1 a V28 e Amount, sem a coluna alvo.

    Retorno:
        pd.DataFrame com 31 colunas, na ordem V1 a V28, log_amount, hour_sin e hour_cos, e o
        mesmo índice de X.

    Exemplo de uso:
        X_train = engineer_features(df_train.drop(columns=["Class"]))
        # Time = 90000 s cai na hora 1 do ciclo: hour_sin ≈ 0.259 e hour_cos ≈ 0.966
        # Amount = 99 resulta em log_amount = ln(100) ≈ 4.605
    """
    # fds-et-19-anomaly-detection-models, célula 11, linha 50
    hour = (X["Time"] % 86400) / 3600
    features = X[V_COLUMNS].copy()
    # fds-et-19-anomaly-detection-models, célula 11, linha 48
    features["log_amount"] = np.log1p(X["Amount"])
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return features
