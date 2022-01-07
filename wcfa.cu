#include <bits/types/FILE.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <thrust/device_vector.h>

#include <iostream>
#include "const.h"
#include "util.h"

void populateMatrix(int* matrix, int rows, int cols, const char* path) {
  FILE* fp = fopen(path, "r");
  int token = readToken(fp, NUM_NOT_FOUND);
  for (int i = 0; token != EOF; ) {
	  if (token != NUM_NOT_FOUND) {
      matrix[i] = token;
      i++;
	  }
	  token = readToken(fp, NUM_NOT_FOUND);
	}
  fclose(fp);
}

void populateStore (int* store, int cols, int lams, int vals) {
  memset(store, -1, vals * cols * sizeof(int));

  for (int i = 0; i < lams; i++) {
    store[i*cols + 0] = 2;
    store[i*cols + 1] = i;
  }
  for (int i = lams; i < vals; i++) {
    store[i*cols + 0] = 1;
  }
}

void adOrAp(int deps[], int row, int value, int rowSize) {
  int index = row * rowSize;
  int numEls = deps[index];
  // fprintf(stdout, "Value = %d, Index = %d, Row = %d, Numels = %d before\n", value, index, row, numEls);
  for (int i = index + 1; i < (index + numEls + 1); i++) {
    // fprintf(stdout, "Checking index %d = %d; value = %d", i, deps[i], value);
    if (deps[i] == value){
      return;
    }
  }
  // fprintf(stdout, "Value = %d, Index = %d, Found = %d after\n", value, index, found);
  deps[index]++;
  deps[index + numEls + 1] = value;
}

void makeDepGraph(int dp[], int ar1[], int ar2[], int cf[], int calls, int storeSize, int rowSize, int vals) {
  // fprintf(stdout, "Rowsize = %d, Storesize = %d\n", rowSize, storeSize);
  memset(dp,NUM_NOT_FOUND, storeSize);
  for (int i = 0; i < vals; i++) {
    dp[i*rowSize] = 0;
  }
  for (int i = 0; i < calls; i++) {
    adOrAp(dp, ar1[i], i, rowSize);
    adOrAp(dp, ar2[i], i, rowSize);
    adOrAp(dp, cf[i], i, rowSize);
  }
}

__device__ void addOrAppend(int * matrix, int key, int value) {
}


__device__ void update(std::set<int> store[],std::queue<int> &workList, std::map<int, std::vector<int>> &deps, int arg, int var, int callsite) {
}

__global__ void runAnalysis(int ar1[], int ar2[], int cf[], int calls, int lams, int vals, int rowSize) {

}

int main(int argc, char** argv) {
  std::string testId = DEFAULT_TEST;

  if (argc == 1){
  } else if (argc == 2){
    testId = argv[1];
  } else {
    return 1;
  }

  std::string outputPath = OUTPUT_DIR + testId;
  std::string paramsPath = INPUT_DIR + testId + PARAMS_PATH;
  std::string arg1Path = INPUT_DIR + testId + ARG1_PATH;
  std::string arg2Path = INPUT_DIR + testId + ARG2_PATH;
  std::string funPath = INPUT_DIR + testId + FUN_PATH;

  int lams, vars, calls;
  {
    FILE *fp = fopen(paramsPath.c_str(), "r");
    fscanf(fp, "%d %d %d\n", &lams, &vars, &calls);
    fclose(fp);
  }

  const int vals = 3 * lams;
  const int rowSize = std::max(lams/10, 10);
  const int storeSize = vals * rowSize * sizeof(int);

  // fprintf (stderr, "Program parameters\n");
  // fprintf (stderr, "------------------\n");
  // fprintf (stderr,
       // "lams: %d\nvars: %d\nvals: %d\ncalls: %d\nterms: %d\n",
       // lams, vars, vals, calls, lams+vars+calls);
  // fprintf (stdout, "Directories: \n Ar1 = %s , Ar2 = %s , Params = %s\n", arg1Path.c_str(), arg2Path.c_str(), paramsPath.c_str());
  // fprintf (stdout, "Directories: \n OUTPUT_DIR = %s , INPUT_DIR = %s , FUN_PATH = %s\n", OUTPUT_DIR, INPUT_DIR, FUN_PATH);

  int *store = (int*)malloc(storeSize);
  int *deps = (int*)malloc(storeSize);
  int *callFun = (int*)malloc(calls * sizeof(int));
  int *callArg1 = (int*)malloc(calls * sizeof(int));
  int *callArg2 = (int*)malloc(calls * sizeof(int));

  // Populate store
  populateStore(store, rowSize, lams, vals);

  // Read in the FUN matrix
  // fprintf(stderr, "Reading CALLFUN (%d x %d) ... ", calls, 1);
  populateMatrix(callFun, calls, 2, funPath.c_str());
  // fprintf(stderr, "Populated FUN\n");
  printMatrix(callFun, 1, calls);

  // Read in the ARG1 matrix
  // fprintf(stderr, "Reading ARG1 (%d x %d) ... ", calls, 1);
  populateMatrix(callArg1, calls, 2, arg1Path.c_str());
  // fprintf(stderr, "Populated ARG1\n");
  printMatrix(callArg1, 1, calls);


  // Read in the ARG2 matrix
  // fprintf(stderr, "Reading ARG2 (%d x %d) ... ", calls, 1);
  populateMatrix(callArg2, calls, 2, arg2Path.c_str());
  // fprintf(stderr, "Populated ARG2\n");
  printMatrix(callArg2, 1, calls);

  // Construct the dependency graph
  // fprintf(stderr, "Constructing dependency graph (%d x %d) ... ", calls, 1);
  makeDepGraph(deps, callArg1, callArg2, callFun, calls, storeSize, rowSize, vals);
  // fprintf(stderr, "Graph constructed\n");
  // printMatrix(deps, vals, rowSize);

  // Move data to the device
  int *dp, *st, *cf, *a1, *a2;
  cudaMalloc((void**)&a1, calls*sizeof(int));
  cudaMalloc((void**)&a2, calls*sizeof(int));
  cudaMalloc((void**)&cf, calls*sizeof(int));
  cudaMalloc((void**)&dp, storeSize);
  cudaMalloc((void**)&st, storeSize);

  cudaMemcpy(a1, callArg1, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(a2, callArg2, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cf, callFun, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dp, deps, storeSize, cudaMemcpyHostToDevice);
  cudaMemcpy(st, store, storeSize, cudaMemcpyHostToDevice);

  // Run the analysis
  runAnalysis<<<NUM_BLOCK, NUM_THREAD>>>(callArg1, callArg2, callFun, calls, lams, vals, rowSize);
  // printStore(store, vals);
  // printDeps(deps);

  // Write out the result
  // fprintf(stderr, "Writing %s\n", outputPath.c_str());
  // FILE* resFp = fopen(outputPath.c_str(), "w");
  // reformatStore(store, vals, resFp);
  // fclose(resFp);

  // Deallocate memory
  cudaFree(a1);
  cudaFree(a2);
  cudaFree(cf);
  cudaFree(dp);
  cudaFree(st);
  
  free(callArg1);
  free(callArg2);
  free(callFun);
  // free(deps);
  // free(store);

  return EXIT_SUCCESS;
}
