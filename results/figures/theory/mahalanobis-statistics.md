# Figuras da distância de Mahalanobis

Gerado por `src/analysis/mahalanobis_figures.py`, com dados simulados (semente 42).

## 01-euclidean-vs-mahalanobis

- 150 pontos de uma normal bivariada com desvios padrão 3 e 1 e correlação 0.8.
- A e B ficam à distância euclidiana 3 do centro.
- Distância de Mahalanobis de A: 1.00.
- Distância de Mahalanobis de B: 5.82.

## 02-classical-vs-robust-covariance

- 60 observações regulares e 10 discrepantes.
- Elipse de tolerância de 97.5%: distância de Mahalanobis até 2.716.
- Discrepantes dentro da elipse clássica: 10 de 10.
- Discrepantes dentro da elipse robusta: 0 de 10.
