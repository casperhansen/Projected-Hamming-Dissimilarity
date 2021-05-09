#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// gcc runTime.c -o test -std=c99 -march=native -O3 -mcmodel=large

int hammingDistance (uint64_t x, uint64_t y) {
        uint64_t res = x ^ (y);
        return __builtin_popcountll (res);
}

int projHamDis (uint64_t x, uint64_t y) {
        uint64_t res = x ^ (y & x);
        return __builtin_popcountll (res);
}


#define users 1
#define items 100000000
#define runs 1000
uint64_t users_array[users];
uint64_t items_array[items];


int to_do_stupid = 0;
int main()
{
  double t_hamming = 0;
  double t_self_mask = 0;
  printf("users: %i, items: %i, runs: %i\n",users,items,runs);
  for(int uc = users; uc != 0; --uc)
  {
    users_array[uc] = (uint64_t) rand();
  }

  for(int ic = items; ic != 0; --ic)
  {
    items_array[ic] = (uint64_t) rand();
  }


  // warm up
  int res;
  for (int run = runs; run != 0; --run)
  {
    to_do_stupid = 0;
    for(int uc = users; uc != 0; --uc)
      {
        for(int ic = items; ic != 0; --ic)
          {
            res = hammingDistance(users_array[uc],items_array[ic]);
            
          }
      }
    to_do_stupid = res ^ to_do_stupid;
  }

  // warm up done

  clock_t t;

  for (int run = runs; run != 0; --run)
  {
    to_do_stupid = 0;
    t = clock();
    for(int uc = users; uc != 0; --uc)
      {
        for(int ic = items; ic != 0; --ic)
          {
            res = hammingDistance(users_array[uc],items_array[ic]);
            
          }
      }
    t = clock() - t;
    to_do_stupid = res ^ to_do_stupid;
    t_hamming += ((double)t)/CLOCKS_PER_SEC;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("Hamming Distance took %f seconds to execute, %d \n", time_taken, run);
  }


  for (int run = runs; run != 0; --run)
  {
    to_do_stupid = 0;
    t = clock();
    for(int uc = users; uc != 0; --uc)
      {
        for(int ic = items; ic != 0; --ic)
          {
            res = projHamDis(users_array[uc],items_array[ic]);
          }
      }
    t = clock() - t;
    to_do_stupid = res ^ to_do_stupid;
    t_self_mask += ((double)t)/CLOCKS_PER_SEC;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("projected Hamming dissimilarity took %f seconds to execute, %d \n", time_taken, run);
  }

  printf("Hammding distance: %f,        projected Hamming dissimilarity: %f\n",t_hamming/((double) runs),t_self_mask/((double) runs));

  return 0;
}