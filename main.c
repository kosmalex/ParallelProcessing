#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_THREADS omp_get_max_threads()
#define OMP_TIME omp_get_wtime()
#define OMP_TID omp_get_thread_num()

#define N 100000
#define M 500

struct data{
  int   max;
  int   min;
  float median;
};

void print_res(struct data src){
  printf("{max: %d, min: %d, median:%.3f}\n", src.max, src.min, src.median);
}

inline void parallel_func(int* A, size_t np);

int main(int argc, char** argv){
  
  srand(time(NULL));
  
  int* A = (int*)calloc(M*N, sizeof(int));
  
  #pragma omp parallel for schedule(guided) num_threads(MAX_THREADS)
  for(size_t i=0; i< M*N; i++) {
    double t = (double)(rand() % 1000) / 1000.0;
    A[i] = (int)(-100 * t + (1-t) * 100);
  }

  struct data res_s = {A[0], A[0], 0};
  
  double time = OMP_TIME;  
  for(size_t i=0; i<N*M; i++){
   int temp = A[i];
   res_s.median += temp;
   res_s.max = (res_s.max <= temp) ? temp : res_s.max;
   res_s.min = (res_s.min >= temp) ? temp : res_s.min;
  }
  res_s.median /= M*N;
  time = OMP_TIME - time;
  
  printf("\n------------- Serial [%d, %d] ----------------\n", N, M);
  printf("Time: %.6f ms\n", time * 1000.0);
  print_res(res_s);
  
  ////////////////////////////////////////
  //                                    //
  //       PARALLEL VERSION             //
  //                                    //
  ////////////////////////////////////////

  for(size_t np=1; np<=MAX_THREADS; np<<=1)
    parallel_func(A, np);

  return 0;
}

void parallel_func(int* A, size_t np){
  double time = OMP_TIME;
  
  // Init tree
  struct data* tree = (struct data*)malloc(MAX_THREADS * sizeof(struct data));
  for(size_t i=0; i<MAX_THREADS; i++) {
    tree[i].max = tree[i].min = A[0];
    tree[i].median = 0;
  }

  struct data per_thread_result;
  size_t tid, i;
  int temp;

  #pragma omp parallel private(tid, temp, per_thread_result, i) num_threads(np)
  {
    tid = OMP_TID;
    per_thread_result = tree[tid];

    #pragma omp for schedule(static)
    for(i=0; i<M*N; i++){
      temp = A[i];

      per_thread_result.median += temp;
      per_thread_result.max = (per_thread_result.max <= temp) ? temp : per_thread_result.max;
      per_thread_result.min = (per_thread_result.min >= temp) ? temp : per_thread_result.min;
    }
    
    /* Initialize the values at the first level of the tree */
    tree[tid] = per_thread_result;

    /* Syncronize threads before entering tree stage */
    #pragma omp barrier
    
    /* The result is gathered to Thread with id 0 with a previoulsy implemented tree*/
    for(int j=2; j<=np; j<<=1){
      if(tid % j == 0){
        //printf(" {j %d} Thread %d is taking from Thread %ld\n", j, tid, tid + j/2);
        tree[tid].median += tree[tid + j/2].median;
        tree[tid].max = (tree[tid].max >= tree[tid + j/2].max) ? tree[tid].max : tree[tid + j/2].max;
        tree[tid].min = (tree[tid].min <= tree[tid + j/2].min) ? tree[tid].min : tree[tid + j/2].min;
      }
      #pragma omp barrier
    }
  }

  /* Last step. Normalize the median */
  tree[0].median /= N*M;

  time = OMP_TIME - time;

  printf("\n\n-------------- Parallel[%ld] [%d, %d] -------------\n", np, N, M);
  printf("Time: %.6f ms\n", time * 1000.0);
  print_res(tree[0]);
}
