#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <sys/time.h>

#define BUFFER_SIZE (sizeof(double) * nplanets)

float tdiff(struct timeval *start, struct timeval *end) {
	return (end->tv_sec - start->tv_sec) +
		   1e-6 * (end->tv_usec - start->tv_usec);
}

struct PlanetData {
	double *mass;
	double *x;
	double *y;
	double *vx;
	double *vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
	seed ^= (seed << 21);
	seed ^= (seed >> 35);
	seed ^= (seed << 4);
	return seed;
}

double randomDouble() {
	unsigned long long next = randomU64();
	next >>= (64 - 26);
	unsigned long long next2 = randomU64();
	next2 >>= (64 - 26);
	return ((next << 27) + next2) / (double)(1LL << 53);
}

int nplanets;
int timesteps;
double dt;
double G;
double EPSILON = 0.0001;

void compute(PlanetData &data) {
	double *x = (double *)alloca(BUFFER_SIZE);
	double *y = (double *)alloca(BUFFER_SIZE);

	memcpy(x, data.x, BUFFER_SIZE);
	memcpy(y, data.y, BUFFER_SIZE);

	for (int t = 0; t < timesteps; t++) {
		for (int i = 0; i < nplanets; i++) {
			double vx_acc = 0;
			double vy_acc = 0;

			for (int j = 0; j < nplanets; j++) {
				double dx = data.x[j] - data.x[i];
				double dy = data.y[j] - data.y[i];
				double distSqr = dx * dx + dy * dy + EPSILON;
				double invDist = data.mass[i] * data.mass[j] / sqrt(distSqr);
				double invDist3 = invDist * invDist * invDist;
				vx_acc += dx * invDist3;
				vy_acc += dy * invDist3;
			}

			data.vx[i] += dt * vx_acc;
			data.vy[i] += dt * vy_acc;

			x[i] += dt * data.vx[i];
			y[i] += dt * data.vy[i];
		}

		std::memcpy(data.x, x, BUFFER_SIZE);
		std::memcpy(data.y, y, BUFFER_SIZE);
	}
}

int main(int argc, const char **argv) {
	if (argc < 2) {
		printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
		return 1;
	}
	nplanets = atoi(argv[1]);
	timesteps = atoi(argv[2]);
	dt = 0.001;
	G = 6.6743;

	PlanetData data;
	data.mass = (double *)malloc(BUFFER_SIZE);
	data.x = (double *)malloc(BUFFER_SIZE);
	data.y = (double *)malloc(BUFFER_SIZE);
	data.vx = (double *)malloc(BUFFER_SIZE);
	data.vy = (double *)malloc(BUFFER_SIZE);

	for (int i = 0; i < nplanets; i++) {
		data.mass[i] = randomDouble() * 10 + 0.2;
		data.x[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
		data.y[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
		data.vx[i] = randomDouble() * 5 - 2.5;
		data.vy[i] = randomDouble() * 5 - 2.5;
	}

	struct timeval start, end;
	gettimeofday(&start, NULL);
	compute(data);
	gettimeofday(&end, NULL);

	double final_x = data.x[nplanets - 1];
	double final_y = data.y[nplanets - 1];
	printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
		   tdiff(&start, &end), final_x, final_y);

	return 0;
}