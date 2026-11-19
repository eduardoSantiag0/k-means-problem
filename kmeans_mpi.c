#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>

#define MASTER if (rank == 0)

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
    int cluster;
} Centroids;

void prepare_scatterv(int n_points, int n_process, int *sendcounts, int *displs);
void reset_local_accumulators(double local_sums[][3], int *local_counts, int k);


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
    int rank, n_process;
    int changed;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);

    // Cada processo possui seu proprio espaco
    double local_sums[k][3];
    double global_sums[k][3];
    int local_counts[k];

    int base = n_points / n_process;
    int remainder = n_points % n_process;

    int local_n = base;
    if (rank < remainder) {
        local_n++;
    }

    // Mesmo sem pontos locais, reservar espaco para um elemento
    int allocation_n = local_n;
    if (allocation_n == 0) {
        allocation_n = 1;
    }

    Points *local_n_points = malloc(allocation_n * sizeof(Points));

    if (!local_n_points) {
        fprintf(stderr, "Error allocating local_n_points\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Apenas o master
    int *sendcounts  = NULL;
    int *displs = NULL;
    MASTER {
        if (n_process > INT_MAX / (int)sizeof(int)) {
            fprintf(stderr, "Too many MPI processes\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        sendcounts = malloc(n_process * sizeof(int));
        displs = malloc(n_process * sizeof(int));

        if (!sendcounts || !displs) {
            fprintf(stderr, "Error allocating Scatterv arrays\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        prepare_scatterv(n_points, n_process, sendcounts, displs);
    }

    MPI_Scatterv(
        points,
        sendcounts,
        displs,
        MPI_BYTE,
        local_n_points,
        local_n * (int)sizeof(Points),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    
    free(sendcounts);
    free(displs);

    while (1) {

        // Reiniciar os acumuladores locais
        reset_local_accumulators(local_sums, local_counts, k);

        // Somar as coordenadas e contar os pontos de cada cluster local
        for (int i = 0; i < local_n; i++) {
            int c = local_n_points[i].cluster;

            local_sums[c][0] += local_n_points[i].x;
            local_sums[c][1] += local_n_points[i].y;
            local_sums[c][2] += local_n_points[i].z;

            local_counts[c]++;
        }

        // Quantos pontos em cada cluster (resultado no rank 0)
        MPI_Reduce(
            local_counts,
            counts,
            k,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        // Soma as coordenadas dos pontos (resultado no rank 0)
        MPI_Reduce(
            local_sums,
            global_sums,
            3 * k,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        MASTER {
            update_centroids(global_sums, counts, centroids, k);
        }

        MPI_Bcast(
            centroids,
            k * (int)sizeof(Centroids),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        int local_changed = assign_clusters(
            local_n_points, centroids, 0, local_n, k
        );

        // Somar quantos pontos mudaram
        MPI_Allreduce(
            &local_changed,
            &changed,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        if (changed == 0) {
            break;
        }

        // Os centroides atualizados serao enviados no inicio da proxima iteracao
    }

    free(local_n_points);
}


int main(int argc, char *argv[]) {
    int rank, n_process;
    double start, end;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Erro em MPI_Init. Terminando...\n");
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);

    int k = 0;
    int n_points = 0;

    // Ler k e n da entrada padrao
    MASTER {
        if (scanf("%d %d", &k, &n_points) != 2 || k <= 0 || n_points <= 0) {
            fprintf(stderr, "Invalid input\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (n_points > INT_MAX / (int)sizeof(Points) ||
            k > INT_MAX / (int)sizeof(Centroids) ||
            k > INT_MAX / 3) 
        {
            fprintf(stderr, "Input too large for MPI counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Centroids *centroids = malloc(k * sizeof(Centroids));
    Points *points = NULL;
    int counts[k];

    MASTER {
        points = malloc(n_points * sizeof(Points));
    }

    if (!centroids || (rank == 0 && !points)) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Lendo os inputs
    MASTER {
        for (int i = 0; i < k; i++) {
            if (scanf("%lf %lf %lf",
                      &centroids[i].x,
                      &centroids[i].y,
                      &centroids[i].z) != 3) {
                fprintf(stderr, "Error reading centroid %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            centroids[i].cluster = i;
        }

        // Ler os N pontos
        for (int i = 0; i < n_points; i++) {
            if (scanf("%lf %lf %lf",
                      &points[i].x,
                      &points[i].y,
                      &points[i].z) != 3) {
                fprintf(stderr, "Error reading point %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            points[i].cluster = -1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    kmeans(points, centroids, n_points, k, counts);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MASTER {
        for (int i = 0; i < k; i++) {
            printf("%.6f %.6f %.6f\n",
                   centroids[i].x,
                   centroids[i].y,
                   centroids[i].z);
        }

        fprintf(stderr, "Tempo = %.9f s\n", end - start);
    }

    free(centroids);
    free(points);

    MPI_Finalize();
    return 0;
}


void prepare_scatterv(int n_points, int n_process, int *sendcounts, int *displs) {
    int base = n_points / n_process;
    int remainder = n_points % n_process;
    int offset = 0;

    for (int i = 0; i < n_process; i++) {
        int amount = base;
        if (i < remainder) {
            amount++;
        }

        sendcounts[i] = amount * (int)sizeof(Points);
        displs[i] = offset * (int)sizeof(Points);
        offset += amount;
    }
}

void reset_local_accumulators(double local_sums[][3], int *local_counts, int k) {
    for (int j = 0; j < k; j++) {
        local_sums[j][0] = 0;
        local_sums[j][1] = 0;
        local_sums[j][2] = 0;
        local_counts[j] = 0;
    }
}

// mpicc -std=c11 -O2 -Wall -Wextra -o kmeans_mpi kmeans_mpi.c
// mpirun -np 2 ./kmeans_mpi < km.in