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
  for (int i = index + 1; i < (index + numEls); i++) {
    if (deps[i] == value){
      return;
    }
  }
  deps[index]++;
  deps[index + numEls] = value;
}

void makeDepGraph(int dp[], int ar1[], int ar2[], int cf[], int calls, int storeSize, int rowSize, int vals) {
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

void update(int st[], int dp[], bool worklist[], int arg, int var, int
        callsite, int rowSize, int wlSize) {
  int idv = var * rowSize;
  int ida = arg * rowSize;
  int sv = st[idv];
  int sa = st[ida];
  int size = sv - 1;
  int * res = (int*)malloc((sv + sa) * sizeof(int));
  memset(res, NUM_NOT_FOUND,(sv + sa) * sizeof(int));

  for (int i = 0; i < sv-1; i++) {
    res[i] = (st + idv)[i+1];
  }
  for (int i = 1; i < sa; i++) {
    bool found = 0;
    for (int j = 0; j < sv; j++) {
      if (res[j] == (st + ida)[i]) {
        found = 1;
        break;
      }
    }
    if (!found) {
    res[size] = (st + ida)[i];
    size ++;
    }
  }
  size++;
  if (size != sv) {
    int ii = 0;
    st[idv] = size;
    for (int i = idv + 1 ; i < idv + size ; i++) {
      st[i] = res[ii];
      ii++;
    }
    int ds = dp[idv];
    for (int i = idv + 1; i<idv + ds; i++) {
      addElem(dp, var, dp[i], rowSize);
      worklist[dp[i]] = 1;
    }
  }
  free(res);
}

void runIter(int st[], int ar1[], int ar2[], int cf[], int dp[], int
        worklist[], bool newWl[], int lams, int vals, int rowSize, int wlSize) {
    for(int i = 0; i < wlSize; i++) {
        int callSite = worklist[i];
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
}
void runAnalysis(int store[], int callArg1[], int callArg2[], int callFun[], int deps[],
        int worklist[], bool newWl[], int calls, int lams,
        int vals, int rowSize) {

    int wlSize = calls;
    int iteration = 0;
    for(int i = 0; i < wlSize; i++) {
        worklist[i] = i;
    }
    while (wlSize != 0) {

        /* Getting the size of the worklist at each iteration */
        fprintf(stdout, "Worklist size %d = %d: ", iteration, wlSize);
        for(int i = 0; i < wlSize; i++) {
            fprintf(stdout, "%d, ", worklist[i]);
        }
        fprintf(stdout, "\n");

        memset(newWl, 0, calls * sizeof(bool));
        runIter(store, callArg1, callArg2, callFun, deps, worklist, newWl, lams,  vals, rowSize, wlSize);
        wlSize = 0;
        for(int i = 0; i < calls; i++) {
            if(newWl[i]) {
                worklist[wlSize] = i;
                wlSize++;
            }
        }
        iteration++;
    }
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


  int *store = (int*)malloc(storeSize);
  int *deps = (int*)malloc(storeSize);
  int *worklist = (int*)malloc(calls * sizeof(int));
  int *callFun = (int*)malloc(calls * sizeof(int));
  int *callArg1 = (int*)malloc(calls * sizeof(int));
  int *callArg2 = (int*)malloc(calls * sizeof(int));
  bool *newWl = (bool*)calloc(calls, calls * sizeof(bool));

  populateStore(store, rowSize, lams, vals);
  populateMatrix(callFun, calls, 2, funPath.c_str());
  populateMatrix(callArg1, calls, 2, arg1Path.c_str());
  populateMatrix(callArg2, calls, 2, arg2Path.c_str());

  makeDepGraph(deps, callArg1, callArg2, callFun, calls, storeSize, rowSize, vals);

  runAnalysis(store, callArg1, callArg2, callFun, deps, worklist, newWl, calls, lams, vals, rowSize);

  /* Write out the result */
  /* fprintf(stderr, "Writing %s\n", outputPath.c_str()); */
  /* FILE* resFp = fopen(outputPath.c_str(), "w"); */
  /* reformatStore(store, vals, rowSize, resFp); */
  /* fclose(resFp); */

  /* Deallocate memory */

  free(callArg1);
  free(callArg2);
  free(callFun);
  free(deps);
  free(store);
  free(newWl);

  return EXIT_SUCCESS;
}
