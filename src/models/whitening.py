"""Branqueamento das variáveis, que torna a distância euclidiana igual à distância de Mahalanobis.

O HDBSCAN* do cuML só aceita a métrica euclidiana. Transformar os dados por uma matriz W com
W·Wᵀ igual à inversa da covariância faz a distância euclidiana entre dois pontos transformados
igual à distância de Mahalanobis entre os originais, de modo que o HDBSCAN*, o GLOSH e o DISCO
calculados sobre os dados transformados são os de um HDBSCAN* com a métrica de Mahalanobis. O
estudo que fundamenta a escolha está em docs/scratch/notes/mahalanobis-distance-study.md.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from sklearn.utils.validation import check_is_fitted

COVARIANCE_ESTIMATORS = ("empirical", "mcd")
# Valores do diagnóstico nos folds internos da base completa (06/11/2026). De 99% a 100% dos
# pontos, a AUPRC da distância ao centro do MCD fica num platô (0,775), e o ganho vem da
# repesagem. Abaixo de 100%, o FastMCD sorteia subconjuntos e chega a ótimos diferentes conforme a
# versão do scikit-learn: com 99%, a AUPRC do fold 1 foi 0,804 na 1.9.1 e 0,744 na 1.6.1. Com
# 100%, a estimativa bruta é a de todos os pontos da amostra, nada é sorteado no ajuste, e as duas
# versões dão a mesma AUPRC nos cinco folds. Uma amostra de 30 mil pontos rende o mesmo que o
# treino inteiro em cerca de um sétimo do tempo.
DEFAULT_SUPPORT_FRACTION = 1.0
DEFAULT_SAMPLE_SIZE = 30_000
# Piso dos autovalores, relativo ao maior: só age numa covariância degenerada. Pisos de 1e-5 em
# diante já apagam parte do sinal das direções de pouca variância.
DEFAULT_EIGEN_FLOOR = 1e-9


def check_whitening_params(covariance, support_fraction, sample_size, eigen_floor):
    """
    Confere os parâmetros do branqueamento, antes do ajuste ou na configuração do pipeline.

    Objetivo:
        Recusa, com uma mensagem que diz o valor esperado, um estimador desconhecido, uma
        fração de pontos do MCD fora de (0, 1], uma amostra menor que dois pontos e um piso
        dos autovalores fora de (0, 1). Assim, um erro de configuração aparece no início da
        execução, e não depois de horas de grade.

    Parâmetros:
        covariance (str): estimador da covariância, "empirical" ou "mcd".
        support_fraction (float): fração de pontos do suporte bruto do MCD.
        sample_size (int): número máximo de pontos do treino usados no ajuste do MCD.
        eigen_floor (float): piso dos autovalores, como fração do maior.

    Retorno:
        None. Levanta ValueError no primeiro parâmetro inválido.

    Exemplo de uso:
        check_whitening_params("mcd", 0.99, 30_000, 1e-9)   # passa em silêncio
    """
    if covariance not in COVARIANCE_ESTIMATORS:
        raise ValueError(f"covariance={covariance!r}: use um de {COVARIANCE_ESTIMATORS}.")
    if not 0 < support_fraction <= 1:
        raise ValueError(f"support_fraction={support_fraction} precisa estar em (0, 1].")
    if int(sample_size) != sample_size or sample_size < 2:
        raise ValueError(f"sample_size={sample_size} precisa ser um inteiro de pelo menos 2.")
    if not 0 < eigen_floor < 1:
        raise ValueError(f"eigen_floor={eigen_floor} precisa estar em (0, 1).")


class MahalanobisWhitener(TransformerMixin, BaseEstimator):
    """
    Branqueia os dados com a covariância do treino, empírica ou robusta (MCD).

    Objetivo:
        O ajuste estima o centro μ e a covariância Σ do treino e, pela autodecomposição
        Σ = V·Λ·Vᵀ, monta W = V·Λ^(-1/2), com cada autovalor limitado por baixo a eigen_floor
        vezes o maior. A transformação devolve z = (x - μ)·W. A distância euclidiana entre dois
        pontos transformados é a distância de Mahalanobis entre eles, e a norma de z é a
        distância de Mahalanobis do ponto ao centro μ. A autodecomposição evita inverter a
        covariância e, com o piso, tolera uma covariância quase singular.

        Com covariance="empirical", μ é a média e Σ a covariância amostral de todo o treino. Com
        covariance="mcd", os dois vêm do MinCovDet do scikit-learn (FastMCD, com a correção e a
        repesagem finais), ajustado numa amostra de até sample_size pontos do treino. A
        covariância empírica é sensível às próprias anomalias que se quer detectar. O MCD parte
        da estimativa do suporte bruto, com support_fraction dos pontos, e a repesagem descarta
        os pontos além do quantil 0,975 da qui-quadrado e reestima μ e Σ com os demais. Com o
        padrão de 100%, a estimativa bruta é a de toda a amostra, e o ganho vem só da repesagem.

        Nada aqui usa rótulos: o ajuste vê só as variáveis do treino, e a validação e o teste
        recebem só transform.

    Parâmetros:
        covariance (str): "empirical" ou "mcd". O padrão é "mcd".
        support_fraction (float): fração de pontos do suporte bruto do MCD, em (0, 1]. Ignorado
            com a covariância empírica.
        sample_size (int): número máximo de pontos do treino no ajuste do MCD. Com mais pontos
            que isso, sorteia a amostra sem reposição. Ignorado com a covariância empírica.
        eigen_floor (float): piso dos autovalores, como fração do maior, em (0, 1).
        random_state (int | None): semente da amostra e do MinCovDet.

    Atributos:
        location_ (np.ndarray): centro μ, com forma (n_variáveis,).
        covariance_ (np.ndarray): covariância Σ estimada, com forma (n_variáveis, n_variáveis).
        components_ (np.ndarray): matriz W, com forma (n_variáveis, n_variáveis).
        eigenvalues_ (np.ndarray): autovalores de Σ antes do piso, em ordem crescente.
        n_floored_ (int): número de autovalores levados ao piso.

    Exemplo de uso:
        whitener = MahalanobisWhitener("mcd", random_state=42)
        Z_train = whitener.fit_transform(X_train)   # covariância estimada só no treino
        Z_val = whitener.transform(X_val)           # mesma transformação na validação
        np.linalg.norm(Z_val, axis=1)               # distância de Mahalanobis ao centro robusto
    """

    def __init__(
        self,
        covariance="mcd",
        support_fraction=DEFAULT_SUPPORT_FRACTION,
        sample_size=DEFAULT_SAMPLE_SIZE,
        eigen_floor=DEFAULT_EIGEN_FLOOR,
        random_state=None,
    ):
        self.covariance = covariance
        self.support_fraction = support_fraction
        self.sample_size = sample_size
        self.eigen_floor = eigen_floor
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Estima o centro e a covariância do treino e monta a matriz de branqueamento.

        Parâmetros:
            X (array-like): treino, com forma (n_pontos, n_variáveis) e mais pontos que
                variáveis.
            y: ignorado, presente pela interface do scikit-learn.

        Retorno:
            O próprio objeto, ajustado.
        """
        check_whitening_params(
            self.covariance, self.support_fraction, self.sample_size, self.eigen_floor
        )
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] <= X.shape[1]:
            raise ValueError(
                f"X tem forma {X.shape}: a covariância exige mais pontos que variáveis."
            )
        if self.covariance == "empirical":
            self.location_ = X.mean(axis=0)
            self.covariance_ = np.cov(X, rowvar=False)
        else:
            rng = np.random.RandomState(self.random_state)
            sample = X
            if len(X) > self.sample_size:
                sample = X[rng.choice(len(X), int(self.sample_size), replace=False)]
            estimator = MinCovDet(
                support_fraction=self.support_fraction, random_state=self.random_state
            ).fit(sample)
            self.location_ = estimator.location_
            self.covariance_ = estimator.covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        floor = eigenvalues.max() * self.eigen_floor
        self.eigenvalues_ = eigenvalues
        self.n_floored_ = int(np.sum(eigenvalues < floor))
        self.components_ = eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))
        return self

    def transform(self, X):
        """
        Centra e branqueia os dados com o ajuste do treino.

        Parâmetros:
            X (array-like): dados com as mesmas variáveis do ajuste.

        Retorno:
            np.ndarray com a mesma forma de X, em que a distância euclidiana é a de Mahalanobis.
        """
        check_is_fitted(self, "components_")
        return (np.asarray(X, dtype=np.float64) - self.location_) @ self.components_
