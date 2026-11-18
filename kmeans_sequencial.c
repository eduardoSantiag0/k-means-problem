
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef struct {
    double x;
    double y;
    double z;
    int cluster;
} Points;

// Calcula a distancia euclidiana ao quadrado
double distance_squared(const Points *a, const Points *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;

    return dx * dx + dy * dy + dz * dz;
}

// PASSO 1: Atribuir cada ponto ao centroide mais proximo
// Retorna quantos pontos mudaram de cluster
int assign_clusters(Points *points, const Points *centroids,
                    int n_points, int k) {

    int changed = 0;

    for (int i = 0; i < n_points; i++) {

        int nearest = 0;
        double min_distance =
            distance_squared(&points[i], &centroids[0]);

        // Encontrar o centroide mais proximo
        for (int j = 1; j < k; j++) {
            double distance =
                distance_squared(&points[i], &centroids[j]);

            if (distance < min_distance) {
                min_distance = distance;
                nearest = j;
            }
        }

        // Verificar se o ponto mudou de cluster
        if (points[i].cluster != nearest) {
            points[i].cluster = nearest;
            changed++;
        }
    }

    return changed;
}

// PASSO 2: Calcular a media de cada cluster
void update_centroids(const Points *points, Points *centroids,
                      int n_points, int k,
                      Points *sums, int *counts) {

    // Zerar somas e contagens
    for (int j = 0; j < k; j++) {
        sums[j].x = 0;
        sums[j].y = 0;
        sums[j].z = 0;
        counts[j] = 0;
    }

    // Somar as coordenadas dos pontos de cada cluster
    for (int i = 0; i < n_points; i++) {
        int cluster = points[i].cluster;

        sums[cluster].x += points[i].x;
        sums[cluster].y += points[i].y;
        sums[cluster].z += points[i].z;

        counts[cluster]++;
    }

    // Atualizar os centroides com a media
    for (int j = 0; j < k; j++) {
        if (counts[j] > 0) {
            centroids[j].x = sums[j].x / counts[j];
            centroids[j].y = sums[j].y / counts[j];
            centroids[j].z = sums[j].z / counts[j];
        }

        // Cluster vazio: manter o centroide anterior
    }
}

// PASSO 3: Repetir ate convergir
void kmeans(Points *points, Points *centroids,
            int n_points, int k,
            Points *sums, int *counts) {

    int changed;

    do {
        changed = assign_clusters(
            points, centroids, n_points, k
        );

        update_centroids(
            points, centroids, n_points, k, sums, counts
        );

    } while (changed > 0);
}

int main(void) {

    int k;
    int n_points;

    // Ler k e n da entrada padrao
    if (scanf("%d %d", &k, &n_points) != 2 ||
        k <= 0 || n_points <= 0) {
        fprintf(stderr, "Invalid input\n");
        return 1;
    }

    // Alocar memoria
    Points *centroids = malloc(k * sizeof(Points));
    Points *points = malloc(n_points * sizeof(Points));
    Points *sums = malloc(k * sizeof(Points));
    int *counts = malloc(k * sizeof(int));

    if (!centroids || !points || !sums || !counts) {
        fprintf(stderr, "Memory allocation failed\n");
        free(centroids);
        free(points);
        free(sums);
        free(counts);
        return 1;
    }

    // Ler os K centroides iniciais
    for (int i = 0; i < k; i++) {
        if (scanf("%lf %lf %lf",
                  &centroids[i].x,
                  &centroids[i].y,
                  &centroids[i].z) != 3) {
            fprintf(stderr, "Error reading centroid %d\n", i);
            free(centroids);
            free(points);
            free(sums);
            free(counts);
            return 1;
        }
    }

    // Ler os N pontos
    for (int i = 0; i < n_points; i++) {
        if (scanf("%lf %lf %lf",
                  &points[i].x,
                  &points[i].y,
                  &points[i].z) != 3) {
            fprintf(stderr, "Error reading point %d\n", i);
            free(centroids);
            free(points);
            free(sums);
            free(counts);
            return 1;
        }

        points[i].cluster = -1;
    }

    double start_time = omp_get_wtime();

    // Executar K-Means
    kmeans(points, centroids, n_points, k, sums, counts);

    double end_time = omp_get_wtime();
    
    double elapsed_time = end_time - start_time;

    // Imprimir os centroides finais
    for (int i = 0; i < k; i++) {
        printf("%.6f %.6f %.6f\n",
               centroids[i].x,
               centroids[i].y,
               centroids[i].z);
    }
    printf("Tempo de execucao: %.9f segundos\n", elapsed_time);
    free(centroids);
    free(points);
    free(sums);
    free(counts);

    return 0;
}


// gcc  -Wall -Wextra kmeans_sequencial.c -fopenmp -lm -o kmeans_sequencial; ./kmeans_sequencial < km_2000.in

// gcc -O2 -Wall -Wextra -fopenmp kmeans_openmp.c -lm -o kmeans_openmp && OMP_NUM_THREADS=4 ./kmeans_openmp < km_source_params.in