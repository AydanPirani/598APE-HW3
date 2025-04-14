#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <sys/time.h>

#include <omp.h>
#include <immintrin.h>

#define BUFFER_SIZE (sizeof(double) * nplanets)
#define THREAD_THRESHOLD 512

#define ALIGNMENT 64
#define OFFSET (sizeof(__m256d) / sizeof(double))
#define PREFETCH_DISTANCE 16

float tdiff(struct timeval *start, struct timeval *end) {
	return (end->tv_sec - start->tv_sec) +
		   1e-6 * (end->tv_usec - start->tv_usec);
}

struct alignas(ALIGNMENT) PlanetData {
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
__m256d EPSILON = _mm256_set1_pd(0.0001);

inline double reduce_add_pd256(__m256d vec) {
    __m128d low = _mm256_castpd256_pd128(vec);
    __m128d high = _mm256_extractf128_pd(vec, 1);
    __m128d sum = _mm_add_pd(low, high); 
    __m128d shuffled = _mm_unpackhi_pd(sum, sum); 
    sum = _mm_add_sd(sum, shuffled); 
    return _mm_cvtsd_f64(sum); 
}


inline void _inner_loop(PlanetData &data, int i, __m256d &vx_acc, __m256d &vy_acc) {
	__m256d x_i = _mm256_set1_pd(data.x[i]);
	__m256d y_i = _mm256_set1_pd(data.y[i]);
	__m256d mass_i = _mm256_set1_pd(data.mass[i]);
	
	for (int j = 0; j < nplanets; j+=OFFSET) {
		_mm_prefetch((const char *)&data.x[j + PREFETCH_DISTANCE], _MM_HINT_T0);
		_mm_prefetch((const char *)&data.y[j + PREFETCH_DISTANCE], _MM_HINT_T0);
		_mm_prefetch((const char *)&data.mass[j + PREFETCH_DISTANCE], _MM_HINT_T0);

		__m256d x_j = _mm256_loadu_pd(&data.x[j]);
		__m256d y_j = _mm256_loadu_pd(&data.y[j]);
		__m256d mass_j = _mm256_loadu_pd(&data.mass[j]);
		
		__m256d dx = _mm256_sub_pd(x_j, x_i);
		__m256d dy = _mm256_sub_pd(y_j, y_i);

		__m256d dx2 = _mm256_mul_pd(dx, dx);
		__m256d dy2 = _mm256_mul_pd(dy, dy);

		__m256d dist2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), EPSILON);
		__m256d dist_sqrt = _mm256_rsqrt14_pd(dist2);

		__m256d invDist = _mm256_mul_pd(_mm256_mul_pd(mass_i, mass_j), dist_sqrt);
		__m256d invDist3 = _mm256_mul_pd(_mm256_mul_pd(invDist, invDist), invDist);
		
		vx_acc = _mm256_fmadd_pd(dx, invDist3, vx_acc);
		vy_acc = _mm256_fmadd_pd(dy, invDist3, vy_acc);
	}
}

inline void update_step(PlanetData &data, double* x, double* y, int i, double dvx, double dvy) {
	data.vx[i] += dt * dvx;
	data.vy[i] += dt * dvy;
	x[i] += dt * data.vx[i];
	y[i] += dt * data.vy[i];
}

inline void _compute_parallel(PlanetData &data, double* x, double* y) {
	for (int t = 0; t < timesteps; t++) {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < nplanets; i++) {
			__m256d vx_acc = _mm256_setzero_pd();
			__m256d vy_acc = _mm256_setzero_pd();
			
			_inner_loop(data, i, vx_acc, vy_acc);

			double dvx = reduce_add_pd256(vx_acc);
			double dvy = reduce_add_pd256(vy_acc);
			update_step(data, x, y, i, dvx, dvy);
		}
	
		std::memcpy(data.x, x, BUFFER_SIZE);
		std::memcpy(data.y, y, BUFFER_SIZE);
	}
} 

inline void _compute_naive(PlanetData &data, double* x, double* y) {
	for (int t = 0; t < timesteps; t++) {
		for (int i = 0; i < nplanets; i++) {
			__m256d vx_acc = _mm256_setzero_pd();
			__m256d vy_acc = _mm256_setzero_pd();
			
			_inner_loop(data, i, vx_acc, vy_acc);

			double dvx = reduce_add_pd256(vx_acc);
			double dvy = reduce_add_pd256(vy_acc);

			update_step(data, x, y, i, dvx, dvy);
		}

		std::memcpy(data.x, x, BUFFER_SIZE);
		std::memcpy(data.y, y, BUFFER_SIZE);
	}
}

inline void compute(PlanetData &data) {
	double* x = (double *)__builtin_alloca_with_align(BUFFER_SIZE, ALIGNMENT);
	double* y = (double *)__builtin_alloca_with_align(BUFFER_SIZE, ALIGNMENT);

	std::memcpy(x, data.x, BUFFER_SIZE);
	std::memcpy(y, data.y, BUFFER_SIZE);

	if (nplanets >= THREAD_THRESHOLD) {
		_compute_parallel(data, x, y);
		return;
	} 
	
	_compute_naive(data, x, y);
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

	if (nplanets % OFFSET != 0) {
		printf("nplanets must be a multiple of %ld\n", OFFSET);
		exit(1);
	}

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