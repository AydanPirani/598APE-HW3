#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <sys/time.h>

float tdiff(struct timeval *start, struct timeval *end) {
	return (end->tv_sec - start->tv_sec) +
		   1e-6 * (end->tv_usec - start->tv_usec);
}

struct Planet {
	double mass;
	double x;
	double y;
	double vx;
	double vy;
};

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

void compute(PlanetData &data) {
	double x[nplanets];
	double y[nplanets];

	std::memcpy(x, data.x, sizeof(double) * nplanets);
	std::memcpy(y, data.y, sizeof(double) * nplanets);

	for (int t = 0; t < timesteps; t++) {

		for (int i = 0; i < nplanets; i++) {
			for (int j = 0; j < nplanets; j++) {
				double dx = data.x[j] - data.x[i];
				double dy = data.y[j] - data.y[i];

				double distSqr = dx * dx + dy * dy + 0.0001;
				double invDist = data.mass[i] * data.mass[j] / sqrt(distSqr);

				double invDist3 = invDist * invDist * invDist;
				data.vx[i] += dt * dx * invDist3;
				data.vy[i] += dt * dy * invDist3;
			}
			x[i] += dt * data.vx[i];
			y[i] += dt * data.vy[i];
		}

		std::memcpy(data.x, x, sizeof(double) * nplanets);
		std::memcpy(data.y, y, sizeof(double) * nplanets);
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

	PlanetData planetData;
	planetData.mass = (double *)malloc(sizeof(double) * nplanets);
	planetData.x = (double *)malloc(sizeof(double) * nplanets);
	planetData.y = (double *)malloc(sizeof(double) * nplanets);
	planetData.vx = (double *)malloc(sizeof(double) * nplanets);
	planetData.vy = (double *)malloc(sizeof(double) * nplanets);

	for (int i = 0; i < nplanets; i++) {
		planetData.mass[i] = randomDouble() * 10 + 0.2;
		planetData.x[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
		planetData.y[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
		planetData.vx[i] = randomDouble() * 5 - 2.5;
		planetData.vy[i] = randomDouble() * 5 - 2.5;
	}

	struct timeval start, end;
	gettimeofday(&start, NULL);
	compute(planetData);
	gettimeofday(&end, NULL);

	double x = planetData.x[nplanets - 1];
	double y = planetData.y[nplanets - 1];

	printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
		   tdiff(&start, &end), x, y);

	return 0;
}