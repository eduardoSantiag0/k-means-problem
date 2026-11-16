## Algoritmo Sequencial

- K: Numero de Clusters
- N: Numero de pontos
- Pontos: x, y, z no espaço

1. Selecionar o numero de K clusters
1. Escolher k pontos aleatoriamente e atribuir os clusters para cada um deles
    - Uma boa escolha irá ajudar a diminuir a quantidade de iterações.
    
1. Para cada ponto, calcular a distância até todos os k centroides.
1. Associar cada ponto ao centroide mais próximo
1. Calcular a média das coordenadas de todos os pontos pertencentes a cada cluster.
1. Iterar até não mudar nada


## Como paralelizar:

### Referencias:

- Parallelization of the K-Means++ Clustering Algorithm : https://www.iieta.org/journals/isi/paper/10.18280/isi.260106