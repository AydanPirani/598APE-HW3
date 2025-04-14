#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>

#define ALIGNMENT 64
#define OFFSET (256/(sizeof(double) * 8))

float tdiff(struct timeval *start, struct timeval *end) {
	return (end->tv_sec - start->tv_sec) +
		   1e-6 * (end->tv_usec - start->tv_usec);
}

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
size_t buffer_size;

struct alignas(ALIGNMENT) PlanetData {
	double *mass;
	double *x;
	double *y;
	double *vx;
	double *vy;
};

#define PREFETCH_DISTANCE 16
#define TILE_SIZE 8

inline double reduce_add_pd256(__m256d vec) {
    __m128d low = _mm256_castpd256_pd128(vec);
    __m128d high = _mm256_extractf128_pd(vec, 1);
    __m128d sum = _mm_add_pd(low, high); 
    __m128d shuffled = _mm_unpackhi_pd(sum, sum); 
    sum = _mm_add_sd(sum, shuffled); 
    return _mm_cvtsd_f64(sum); 
}


void compute(PlanetData &data) {
	double* x = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);
	double* y = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);

	std::memcpy(x, data.x, sizeof(double) * nplanets);
	std::memcpy(y, data.y, sizeof(double) * nplanets);

	for (int t = 0; t < timesteps; t++) {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < nplanets; i++) {
			double xi = x[i];
			double yi = y[i];
			double vxi = data.vx[i];
			double vyi = data.vy[i];
			double massi = data.mass[i];
			
			__m256d xi_vec = _mm256_set1_pd(xi);
			__m256d yi_vec = _mm256_set1_pd(yi);
			__m256d massi_vec = _mm256_set1_pd(massi);
			__m256d dt_vec = _mm256_set1_pd(dt);
			__m256d epsilon_vec = _mm256_set1_pd(0.0001);
			
			__m256d acc_vx = _mm256_setzero_pd();
			__m256d acc_vy = _mm256_setzero_pd();
			
			for (int j = 0; j < nplanets; j+=OFFSET) {
				_mm_prefetch((char*)&data.x[j+PREFETCH_DISTANCE], _MM_HINT_T0);
				_mm_prefetch((char*)&data.y[j+PREFETCH_DISTANCE], _MM_HINT_T0);
				_mm_prefetch((char*)&data.mass[j+PREFETCH_DISTANCE], _MM_HINT_T0);
				__m256d xj = _mm256_loadu_pd(&data.x[j]);
				__m256d yj = _mm256_loadu_pd(&data.y[j]);
				__m256d massj = _mm256_loadu_pd(&data.mass[j]);
				
				__m256d dx = _mm256_sub_pd(xj, xi_vec);
				__m256d dy = _mm256_sub_pd(yj, yi_vec);

				__m256d dx2 = _mm256_mul_pd(dx, dx);
				__m256d dy2 = _mm256_mul_pd(dy, dy);

				__m256d dist2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), epsilon_vec);
				__m256d dist = _mm256_rsqrt14_pd(dist2);
				// __m256d dist = _mm256_sqrt_pd(dist2);

				__m256d invDist = _mm256_mul_pd(_mm256_mul_pd(massi_vec, massj), dist);
				__m256d invDist3 = _mm256_mul_pd(_mm256_mul_pd(invDist, invDist), invDist);
				
				__m256d dvx = _mm256_mul_pd(dx, invDist3);
				__m256d dvy = _mm256_mul_pd(dy, invDist3);
				
				acc_vx = _mm256_add_pd(acc_vx, dvx);
				acc_vy = _mm256_add_pd(acc_vy, dvy);
			}
				
			vxi += reduce_add_pd256(_mm256_mul_pd(acc_vx, dt_vec));
            vyi += reduce_add_pd256(_mm256_mul_pd(acc_vy, dt_vec));
			
			data.vx[i] = vxi;
			x[i] += dt * vxi;
			
			data.vy[i] = vyi;
			y[i] += dt * vyi;
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

	if (nplanets % 8 != 0) {
		printf("nplanets must be a multiple of 8\n");
		return 1;
	}

	dt = 0.001;
	G = 6.6743;
	buffer_size = sizeof(double) * nplanets;

	PlanetData planetData;
	planetData.mass = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);
	planetData.x = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);
	planetData.y = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);
	planetData.vx = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);
	planetData.vy = (double*) __builtin_alloca_with_align(buffer_size, ALIGNMENT);

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