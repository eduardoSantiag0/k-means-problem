#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    double x;
    double y;
    double z;
    int cluster;
} Points;

typedef struct {
    double x;
    double y;
    double z;
} Centroids;

void reset_local_accumulators(double local_sums[][3], int *local_counts, int k);

// Calcula a distancia euclidiana ao quadrado
double distance_squared(const Points *a, const Centroids *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;

    return dx * dx + dy * dy + dz * dz;
}

// PASSO 1: Atribuir cada ponto ao centroide mais proximo
// Retorna quantos pontos mudaram de cluster
int assign_clusters(Points *points, const Centroids *centroids,
                    int start, int end, int k) {

    int changed = 0;

    for (int i = start; i < end; i++) {
        int nearest = 0;
        double min_distance = distance_squared(&points[i], &centroids[0]);

        // Encontrar o centroide mais proximo
        for (int j = 1; j < k; j++) {
            double distance = distance_squared(&points[i], &centroids[j]);

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
void update_centroids(double global_sums[][3], int *counts,
                      Centroids *centroids, int k) {

    for (int j = 0; j < k; j++) {
        if (counts[j] > 0) {
            centroids[j].x = global_sums[j][0] / counts[j];
            centroids[j].y = global_sums[j][1] / counts[j];
            centroids[j].z = global_sums[j][2] / counts[j];
        }
        // Se o cluster estiver vazio, manter o centroide anterior
    }
}

// PASSO 3: Repetir ate convergir
void kmeans(Points *points, Centroids *centroids,
            int n_points, int k, int *counts) {

    int changed = 0;
    int n_threads = omp_get_max_threads();

    if (k > INT_MAX / n_threads) {
        fprintf(stderr, "Too many accumulators\n");
        exit(EXIT_FAILURE);
    }

    double (*partial_sums)[3] =
        calloc(n_threads * k, sizeof(*partial_sums));
    int *partial_counts =
        calloc(n_threads * k, sizeof(int));
    double (*global_sums)[3] =
        calloc(k, sizeof(*global_sums));

    if (!partial_sums || !partial_counts || !global_sums) {
        fprintf(stderr, "Memory allocation failed\n");
        free(partial_sums);
        free(partial_counts);
        free(global_sums);
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        int start = (long long)tid * n_points / total;
        int end = (long long)(tid + 1) * n_points / total;

        double (*local_sums)[3] = &partial_sums[tid * k];
        int *local_counts = &partial_counts[tid * k];

        while (1) {

            // PASSO 1: Reiniciar changed
            #pragma omp single
            {
                changed = 0;
            }

            // PASSO 2: Atribuir pontos aos clusters
            int local_changed = assign_clusters(
                points, centroids, start, end, k
            );

            // PASSO 3: Somar mudanças de todas as threads
            #pragma omp atomic update
            changed += local_changed;

            #pragma omp barrier

            // PASSO 4: Verificar convergência
            if (changed == 0) {
                break;
            }

            // PASSO 5: Reiniciar acumuladores locais
            reset_local_accumulators(local_sums, local_counts, k);

            // PASSO 6: Acumular coordenadas e contagens
            for (int i = start; i < end; i++) {
                int c = points[i].cluster;

                local_sums[c][0] += points[i].x;
                local_sums[c][1] += points[i].y;
                local_sums[c][2] += points[i].z;

                local_counts[c]++;
            }

            #pragma omp barrier

            // PASSO 7: Combinar resultados das threads
            // PASSO 8: Atualizar centroides
            #pragma omp single
            {
                for (int j = 0; j < k; j++) {
                    global_sums[j][0] = 0;
                    global_sums[j][1] = 0;
                    global_sums[j][2] = 0;
                    counts[j] = 0;

                    for (int t = 0; t < total; t++) {
                        int pos = t * k + j;

                        global_sums[j][0] += partial_sums[pos][0];
                        global_sums[j][1] += partial_sums[pos][1];
                        global_sums[j][2] += partial_sums[pos][2];
                        counts[j] += partial_counts[pos];
                    }
                }

                update_centroids(
                    global_sums, counts, centroids, k
                );
            }
            // Barreira implícita do single
        }
    }

    free(partial_sums);
    free(partial_counts);
    free(global_sums);
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

    Centroids *centroids = malloc(k * sizeof(Centroids));
    Points *points = malloc(n_points * sizeof(Points));
    int counts[k];

    if (!centroids || !points) {
        fprintf(stderr, "Memory allocation failed\n");
        free(centroids);
        free(points);
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
            return 1;
        }

        points[i].cluster = -1;
    }

    double t0 = omp_get_wtime();

    kmeans(points, centroids, n_points, k, counts);

    double t1 = omp_get_wtime();

    // Imprimir os centroides finais
    for (int i = 0; i < k; i++) {
        printf("%.6f %.6f %.6f\n",
               centroids[i].x,
               centroids[i].y,
               centroids[i].z);
    }

    fprintf(stderr, "Tempo = %.9f s\n", t1 - t0);

    free(centroids);
    free(points);

    return 0;
}

// Funcoes auxiliares
void reset_local_accumulators(double local_sums[][3],
                              int *local_counts, int k) {
    for (int j = 0; j < k; j++) {
        local_sums[j][0] = 0;
        local_sums[j][1] = 0;
        local_sums[j][2] = 0;
        local_counts[j] = 0;
    }
}

// gcc -std=c11 -O2 -Wall -Wextra -fopenmp -o kmeans_openmp kmeans_openmp_padronizado.c
// OMP_NUM_THREADS=4 ./kmeans_openmp < km_source_params.in