#include <bits/types/FILE.h>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include "const.h"
#include "util.h"

void populateMatrix(int matrix[], int rows, int cols, const char path[]) {
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

void populateStore (int store[], int cols, int lams, int vals) {
  memset(store, NUM_NOT_FOUND, vals * cols * sizeof(int));

  for (int i = 0; i < lams; i++) {
    store[i*cols + 0] = 2;
    store[i*cols + 1] = i;
  }
  for (int i = lams; i < vals; i++) {
    store[i*cols + 0] = 1;
  }
}

void addElem(int deps[], int row, int value, int rowSize) {
  int index = row * rowSize;
  int numEls = deps[index];
  // fprintf(stdout, "Value = %d, Index = %d, Row = %d, Numels = %d before\n", value, index, row, numEls);
  for (int i = index + 1; i < (index + numEls); i++) {
    // fprintf(stdout, "Checking index %d = %d; value = %d", i, deps[i], value);
    if (deps[i] == value){
      return;
    }
  }
  // fprintf(stdout, "Value = %d, Index = %d, Found = %d after\n", value, index, found);
  deps[index]++;
  deps[index + numEls] = value;
}

void makeDepGraph(int dp[], int ar1[], int ar2[], int cf[], int calls, int storeSize, int rowSize, int vals) {
  // fprintf(stdout, "Rowsize = %d, Storesize = %d\n", rowSize, storeSize);
  memset(dp,NUM_NOT_FOUND, storeSize);
  for (int i = 0; i < vals; i++) {
    dp[i*rowSize] = 1;
  }
  for (int i = 0; i < calls; i++) {
    addElem(dp, ar1[i], i, rowSize);
    addElem(dp, ar2[i], i, rowSize);
    addElem(dp, cf[i], i, rowSize);
  }
}

__device__ int dAddElem(int matrix[], int row, int value, int rowSize) {
  int index = row * rowSize;
  int numEls = matrix[index];
  for (int i = index + 1; i < (index + numEls + 1); i++) {
    // fprintf(stdout, "Checking index %d = %d; value = %d", i, deps[i], value);
    if (matrix[i] == value || matrix[i] == NUM_NOT_FOUND){
      return 1;
    }
  }
  if (atomicCAS(&matrix[index], numEls, numEls+1)) {
    matrix[index + numEls + 1] = value;
    return 0;
  }
  return 1;
}

__device__ void update(int st[], int dp[], bool worklist[], int arg, int var, int
        callsite, int rowSize, int wlSize) {
  int idv = var * rowSize;
  int ida = arg * rowSize;
  int sv = st[idv];
  int sa = st[ida];
  int size = sv - 1;
  int * res = (int*)malloc((sv + sa) * sizeof(int));
  memset(res, NUM_NOT_FOUND,(sv + sa) * sizeof(int));

  // fprintf(stdout, "Updating: var and arg rows\n");
  // printArray(st + idv, rowSize);
  // printArray(st + ida, rowSize);
  // fprintf(stdout, "Res at the start\n");
  // printArray(res, size);
  for (int i = 0; i < sv-1; i++) {
    res[i] = (st + idv)[i+1];
  }
  // fprintf(stdout, "Res after inserting value vector\n");
  // printArray(res, size);
  for (int i = 1; i < sa; i++) {
    bool found = 0;
    for (int j = 0; j < sv; j++) {
      if (res[j] == (st + ida)[i]) {
        found = 1;
        break;
      }
    }
    if (!found) {
    // fprintf(stdout, "Adding %d on %dth index", (st + ida)[i], size);
    res[size] = (st + ida)[i];
    size ++;
    }
  }
  size++;
  if (size != sv) {
    int ii = 0;
    // fprintf(stdout, "Update happened -> size before %d and after %d\n", sv, size);
    st[idv] = size;
    for (int i = idv + 1 ; i < idv + size ; i++) {
      st[i] = res[ii];
      ii++;
    }
    int ds = dp[idv];
    // fprintf(stdout, "Adding row to worklist:\n");
    // printArray(dp +idv, ds);
    for (int i = idv + 1; i<idv + ds; i++) {
      // fprintf(stdout, "Adding elem %d to row %d\n", dp[i], var);
      // printArray(dp + idv, ds);
      dAddElem(dp, var, dp[i], rowSize);
      worklist[dp[i]] = 1;
    }
  }
  // fprintf(stdout, "Updated: var and arg rows\n");
  // printArray(res, size -1);
  // printArray(st + idv, size);
  // printArray(st + ida, sa);
  free(res);
}

__device__ void handleSite(int st[], int ar1[], int ar2[], int cf[], int dp[],
        int lams, int vals, int rowSize, int callSite, bool newWl[], int wlSize) {
    int fun = cf[callSite];
    int arg1 = ar1[callSite];
    int arg2 = ar2[callSite];
    int idf = fun * rowSize;
    int nef = st[idf];

    for(int i = idf + 1; i < idf + nef; i++) {
      int var1 = st[i] + lams;
      int var2 = st[i] + 2 * lams;

      update(st, dp, newWl, arg1, var1, callSite, rowSize, wlSize);
      update(st, dp, newWl, arg2, var2, callSite, rowSize, wlSize);
    }
}


__device__ unsigned int headDev = 0;

__global__ void runIter(int st[], int ar1[], int ar2[], int cf[], int dp[], int
        worklist[], bool newWl[], int lams, int vals, int rowSize, int wlSize) {
    /* Run the current iteration of the worklist, distribute the callsites */
    	// warp-wise head index of tasks in a block
	__shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

	volatile unsigned int& headWarp = headBlock[threadIdx.y];
	do
	{
		// let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
		if (threadIdx.x == 0) {
			headWarp = atomicAdd(&headDev, WARP_SIZE);

		}
		// task index per thread in a warp
		unsigned int taskIdx = headWarp + threadIdx.x;

		if (taskIdx >= wlSize) {
			return;
		}

		handleSite(st, ar1, ar2, cf, dp, lams, vals, rowSize, worklist[taskIdx], newWl,
                wlSize);
	} while (true);
}

void runAnalysis(int st[], int ar1[], int ar2[], int cf[], int dp[], int worklist[], int calls, int lams,
        int vals, int rowSize) {
    int wlSize = calls;
    bool *newWl = (bool*)malloc(calls);

    for(int i = 0; i < wlSize; i++) {
        worklist[i] = i;
    }
    while (wlSize != 0) {
        memset(newWl, 0, calls * sizeof(bool));
        runIter<<<NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK)>>>(st, ar1, ar2, cf, dp, worklist, newWl, lams, vals,
                rowSize, wlSize);
        cudaDeviceSynchronize();
        wlSize = 0;
        headDev = 0;
        for(int i = 0; i < calls; i++) {
            if(newWl[i]) {
                worklist[wlSize] = i;
                wlSize++;
            }
        }
    }
    free(newWl);
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
  // printMatrix(callFun, 1, calls);

  // Read in the ARG1 matrix
  // fprintf(stderr, "Reading ARG1 (%d x %d) ... ", calls, 1);
  populateMatrix(callArg1, calls, 2, arg1Path.c_str());
  // fprintf(stderr, "Populated ARG1\n");
  // printMatrix(callArg1, 1, calls);


  // Read in the ARG2 matrix
  // fprintf(stderr, "Reading ARG2 (%d x %d) ... ", calls, 1);
  populateMatrix(callArg2, calls, 2, arg2Path.c_str());
  // fprintf(stderr, "Populated ARG2\n");
  // printMatrix(callArg2, 1, calls);

  // Construct the dependency graph
  // fprintf(stderr, "Constructing dependency graph (%d x %d) ... ", calls, 1);
  makeDepGraph(deps, callArg1, callArg2, callFun, calls, storeSize, rowSize, vals);
  // fprintf(stderr, "Graph constructed\n");
  // printMatrix(deps, vals, rowSize);

  // Move data to the device
  int *dp, *st, *cf, *a1, *a2, *worklist;
  cudaMalloc((void**)&a1, calls*sizeof(int));
  cudaMalloc((void**)&a2, calls*sizeof(int));
  cudaMalloc((void**)&cf, calls*sizeof(int));
  cudaMalloc((void**)&worklist, calls*sizeof(int));
  cudaMalloc((void**)&dp, storeSize);
  cudaMalloc((void**)&st, storeSize);

  cudaMemcpy(a1, callArg1, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(a2, callArg2, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cf, callFun, calls*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dp, deps, storeSize, cudaMemcpyHostToDevice);
  cudaMemcpy(st, store, storeSize, cudaMemcpyHostToDevice);

  cudaMemset(worklist, NUM_NOT_FOUND, calls*sizeof(int));
  // Run the analysis
  // runAnalysis(a1, a2, cf, calls, lams, vals, rowSize);
  // printMatrix(deps, vals, rowSize);
  // runAnalysis(store, callArg1, callArg2, callFun, deps, calls, lams, vals, rowSize);
  /* runAnalysis<<<NUM_BLOCK, NUM_THREAD>>>(st, a1, a2, cf, dp, calls, lams, vals, rowSize); */
  runAnalysis(st, a1, a2, cf, dp, worklist, calls, lams, vals, rowSize);
  // printMatrix(store, vals, rowSize);
  // printMatrix(deps, vals, rowSize);

  // cudaMemcpy(callArg1, a1, calls*sizeof(int), cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  // printMatrix(callArg1, 1, calls);

  // Write out the result
  fprintf(stderr, "Writing %s\n", outputPath.c_str());
  FILE* resFp = fopen(outputPath.c_str(), "w");
  reformatStore(store, vals, rowSize, resFp);
  fclose(resFp);

  // Deallocate memory
  cudaFree(a1);
  cudaFree(a2);
  cudaFree(cf);
  cudaFree(dp);
  cudaFree(st);
  cudaFree(worklist);

  free(callArg1);
  free(callArg2);
  free(callFun);
  free(deps);
  free(store);

  return EXIT_SUCCESS;
}
