
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

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


int main( int argc, char *argv[]){

    int numtasks, rank, status;
    status = MPI_Init(&argc,&argv);

	if (status != MPI_SUCCESS) {
		printf ("Erro em MPI_Init. Terminando...\n");
		MPI_Abort(MPI_COMM_WORLD, status);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_sum = rank + 1;
    int global_sum = 0;

    MPI_Reduce(&local_sum,
            &global_sum,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);

    printf("Rank %d: soma = %d\n",
        rank, global_sum);
}


// gcc  -Wall -Wextra kmeans_mpi.c -fopenmp -lm -o kmeans_mpi;

// mpirun -np 4 kmeans_mpi // executa o programa com 4 processos