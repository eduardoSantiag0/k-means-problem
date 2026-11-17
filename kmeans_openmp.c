
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double x;
    double y;
    double z;
    int cluster;
} Points;

// Calcula a distancia euclidiana ao quadrado
double distance_squared(Points a, Points b) {
    double distance = sqrt(
        pow(a.x - b.x, 2) +
        pow(a.y - b.y, 2) +
        pow(a.z - b.z, 2)
    );

    return distance;
}

// PASSO 1: Atribuir cada ponto ao centroide mais proximo
// Retorna quantos pontos mudaram de cluster
int assign_clusters(Points *points, const Points *centroids, int start, int end, 
                int k) {

    int changed = 0;

    for (int i = start; i < end; i++) {

        int nearest = 0;
        double min_distance =
            distance_squared(points[i], centroids[0]);

        // Encontrar o centroide mais proximo
        for (int j = 1; j < k; j++) {
            double distance =
                distance_squared(points[i], centroids[j]);

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


// PASSO 3: Repetir ate convergir
void kmeans(Points *points, Points *centroids,
            int n_points, int k,
            Points *sums, int *counts) {

    int changed;
    
    int N_THREADS = omp_get_max_threads();

    // Uma região de resultados por thread
    Points *partial_sums = calloc(
        N_THREADS * k, sizeof(Points)
    );

    int *partial_counts = calloc(
        N_THREADS * k, sizeof(int)
    );


    if (!partial_sums || !partial_counts) {
        fprintf(stderr, "Fail mem allocation\n");
        free(partial_sums);
        free(partial_counts);
        exit(EXIT_FAILURE);
    }


    // todas as threads modificarão os mesmos centroids, sums e counts. 

    // Precisará de acumuladores locais por thread
        // e uma etapa de redução antes de atualizar os centroides globais.
    #pragma omp parallel num_threads(N_THREADS)
    {

        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        int start = (long long) tid * n_points / total;
        int end = (long long) (tid + 1) * n_points / total;

        // Cada thread possui seu próprio espaço
        Points *local_sums = &partial_sums[tid * k];
        int *local_counts = &partial_counts[tid * k];

        while (1)
        {
            #pragma omp master
            {
                changed = 0;
            }
    
            
            // Todas começarem daqui
            #pragma omp barrier

            int local_changed = assign_clusters(
                points, centroids, start, end, k
            );

    
            #pragma omp atomic
            changed += local_changed;

            // Reinicia o escaço de cada thread
            for (int j = 0; j < k; j++) {
                local_sums[j].x = 0;
                local_sums[j].y = 0;
                local_sums[j].z = 0;
                local_counts[j] = 0;
            }


            for (int i = start; i < end; i++) {
                int c = points[i].cluster;

                local_sums[c].x += points[i].x;
                local_sums[c].y += points[i].y;
                local_sums[c].z += points[i].z;

                local_counts[c]++;
            }

            #pragma omp barrier

            #pragma omp master
            {
                // PASSO 2: Calcular a media de cada cluster
                for (int j = 0; j < k; j++) {
                    sums[j].x = 0;
                    sums[j].y = 0;
                    sums[j].z = 0;
                    counts[j] = 0;

                    for (int t = 0; t < total; t++) {
                        int pos = t * k + j;

                        sums[j].x += partial_sums[pos].x;
                        sums[j].y += partial_sums[pos].y;
                        sums[j].z += partial_sums[pos].z;

                        counts[j] += partial_counts[pos];
                    }

                    if (counts[j] > 0) {
                        centroids[j].x =
                            sums[j].x / counts[j];

                        centroids[j].y =
                            sums[j].y / counts[j];

                        centroids[j].z =
                            sums[j].z / counts[j];
                    }
                }
            }


            #pragma omp barrier


            if (changed == 0) {
                break;
            }
        }

    }

    free(partial_sums);
    free(partial_counts);
    
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
            printf("Point: %lf %lf %lf", points[i].x, points[i].y, points[i].z);
            free(centroids);
            free(points);
            free(sums);
            free(counts);
            return 1;
        }

        points[i].cluster = -1;
    }

    double t0 = omp_get_wtime();

    // Executar K-Means
    kmeans(points, centroids, n_points, k, sums, counts);

    double t1 = omp_get_wtime();

    // Imprimir os centroides finais
    for (int i = 0; i < k; i++) {
        printf("%.6f %.6f %.6f\n",
               centroids[i].x,
               centroids[i].y,
               centroids[i].z);
    }

    printf("Tempo = %.9f s\n", t1 - t0);    
    
    free(centroids);
    free(points);
    free(sums);
    free(counts);

    return 0;
}


// gcc kmeans_openmp.c -lm -fopenmp -o kmeans_openmp && ./kmeans_openmp < km_source_params.in
// OMP_NUM_THREADS=4 ./kmeans_openmp < km_source_params.in