//Constants for the implementation
#ifndef CONST_H

#define NUM_SM 4 // no. of streaming multiprocessors
#define NUM_WARP_PER_SM 48 // maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM 8 // maximum no. of resident blocks per SM
#define NUM_BLOCK NUM_SM * NUM_BLOCK_PER_SM
#define NUM_WARP_PER_BLOCK NUM_WARP_PER_SM / NUM_BLOCK_PER_SM
#define WARP_SIZE 32
#define NUM_NOT_FOUND -42

#define INPUT_DIR "./tests/"
#define OUTPUT_DIR "./res/"

#define DEFAULT_TEST "smallest"
#define PARAMS_PATH "/program-params.lst"
#define FUN_PATH "/Fun.mat"
#define ARG1_PATH "/Arg1.mat"
#define ARG2_PATH "/Arg2.mat"

#endif
