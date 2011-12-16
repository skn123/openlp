#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define ELEM float 

#include "f2c.h"

#define INCOMPLETE 1
#define COMPLETE 0
#define ERROR -1
#define UNBOUNDED -2

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

//#define PRINT_MATRICES

#define USE_OPENCL

typedef struct App {
  //Foundational parts
  ELEM *A;   // Our main data matrix

  ELEM *A_B; // The current basis of the data matrix
  cl_mem cl_A_B;
  cl_mem cl_A_B_tmp;

  ELEM *A_N; // The current independent section of the data matrix
  cl_mem cl_A_N;

  ELEM *B;   // The current row coefficients
  cl_mem cl_B;

  ELEM *C;   // Objective column
  cl_mem cl_C;

  ELEM *C_B; // The basis side of the objective function
  cl_mem cl_C_B;

  ELEM *C_N; // The independent side of the objective function
  cl_mem cl_C_N;

  int NUM_ROWS;   // The total number of rows of A
  int NUM_COLS;   // The total number of cols of A

  int *indices_B; // Indices of the current basis columns
  cl_mem cl_indices_B;

  int *indices_N; // Indices of the current independent columns
  cl_mem cl_indices_N;

  int NUM_B_INDICES; // Helpful quick lookup for number of basic variables
  int NUM_N_INDICES; // Helpful quick lokoup for number of independent variables

  //Helpful temporaries
  ELEM *A_B_trans; // The transposed A_B
  cl_mem cl_A_B_trans;
  cl_mem cl_A_B_trans_tmp;

  ELEM *C_B_trans; // The transposed C_B
  cl_mem cl_C_B_trans;

  ELEM *Y_B;       // The intermediate solve of A_B' \ cB'
  cl_mem cl_Y_B;
  
  ELEM *Y_B_trans; // The transposed Y_B
  cl_mem cl_Y_B_trans;

  ELEM *zRow;      // Temporary calculation of the current objective

  ELEM *Ad;        // Entering variable temporary
  cl_mem cl_Ad;

  ELEM *tVec;      // Leaving variable temporary
  cl_mem cl_tVec;

  ELEM *s1;        // The new coefficients after pivot  
  cl_mem cl_s1;

  ELEM *max_values;
  cl_mem cl_max_values;
  
  int *max_positions;
  cl_mem cl_max_positions;

  ELEM curObj;  // The current objective value
  ELEM newObj;  // The new objective value at the end of this pivot
  
  cl_context context;
  cl_command_queue command_queue;

  cl_kernel negate_kernel;
  cl_kernel max_kernel;
  cl_kernel inverse_kernel;
  cl_kernel multiply_kernel;
  cl_kernel transpose_kernel;
  cl_kernel pairwise_kernel;
} App;

void split_matrix(const ELEM* input, ELEM* outputB, ELEM* outputN, const int* indicesB, const int* indicesN, 
                 const int sizeM, const int sizeN, const int sizeP) {
  int i;
  for (i = 0; i < sizeM; ++i) {
    memcpy(&outputB[sizeP * i], &input[sizeP * indicesB[i]], sizeof(ELEM) * sizeP);
  }
  for (i = 0; i < sizeN; ++i) {
    memcpy(&outputN[sizeP * i], &input[sizeP * indicesN[i]], sizeof(ELEM) * sizeP);
  }
}

void transpose_matrix(const ELEM* input, ELEM* output, const int rows, const int cols) {
  int i, j;
  for (i = 0; i < cols; ++i) {
    for (j = 0; j < rows; ++j) {
      output[i + j * rows] = input[j + i * rows];
    }
  }
}

void cl_transpose_matrix(App *app, const ELEM* input, cl_mem input_in_cl, ELEM *output, cl_mem output_in_cl, const int rows, const int cols) {
  cl_int ret;
  cl_int per_thread = rows;
  size_t localSize = 1;
  size_t globalSize = cols;

  ret = clEnqueueWriteBuffer(app->command_queue, input_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input, 0, NULL, NULL);
  ret = clSetKernelArg(app->transpose_kernel, 0, sizeof(cl_mem), (void*)&input_in_cl);
  ret = clSetKernelArg(app->transpose_kernel, 1, sizeof(cl_mem), (void*)&output_in_cl);
  ret = clSetKernelArg(app->transpose_kernel, 2, sizeof(cl_int), (void*)&per_thread);
  ret = clEnqueueNDRangeKernel(app->command_queue, app->transpose_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, output_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), output, 0, NULL, NULL);
}

void solve_matrix(ELEM* input, ELEM* inout, const int sizeM, const int sizeP) {
  int lda, ldb, b, n, nrhs;
  int *ipiv;

  n = sizeM;
  nrhs = sizeP;
  lda = n;
  ldb = n;
  ipiv = malloc(sizeof(int) * n);

  lapack_int info;

  info = LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, input, lda, ipiv, inout, ldb);
}

void cl_solve_matrix(App *app, ELEM* input1, cl_mem input1_in_cl, cl_mem input1_in_cl_tmp, ELEM* input2, cl_mem input2_in_cl, 
  ELEM* output, cl_mem output_in_cl, const int rows, const int cols) {
  
  cl_int ret;
  cl_int per_thread = rows;
  size_t localSize = 1;
  size_t globalSize_inverse = 1;
  size_t globalSize_multiply = cols;

  ret = clEnqueueWriteBuffer(app->command_queue, input1_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input1, 0, NULL, NULL);
  ret = clSetKernelArg(app->transpose_kernel, 0, sizeof(cl_mem), (void*)&input1_in_cl);
  ret = clSetKernelArg(app->transpose_kernel, 1, sizeof(cl_mem), (void*)&input1_in_cl_tmp);
  ret = clSetKernelArg(app->transpose_kernel, 2, sizeof(cl_int), (void*)&per_thread);
  ret = clEnqueueNDRangeKernel(app->command_queue, app->inverse_kernel, 1, NULL, &globalSize_inverse, &localSize, 0, NULL, NULL);

  ret = clEnqueueWriteBuffer(app->command_queue, input2_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input2, 0, NULL, NULL);
  ret = clSetKernelArg(app->transpose_kernel, 0, sizeof(cl_mem), (void*)&input1_in_cl_tmp);
  ret = clSetKernelArg(app->transpose_kernel, 1, sizeof(cl_mem), (void*)&input2_in_cl);
  ret = clSetKernelArg(app->transpose_kernel, 2, sizeof(cl_mem), (void*)&output_in_cl);
  ret = clSetKernelArg(app->transpose_kernel, 3, sizeof(cl_mem), (void*)&rows);
  ret = clSetKernelArg(app->transpose_kernel, 4, sizeof(cl_mem), (void*)&cols); // 1?
  ret = clSetKernelArg(app->transpose_kernel, 5, sizeof(cl_mem), (void*)&rows);
  ret = clSetKernelArg(app->transpose_kernel, 6, sizeof(cl_mem), (void*)&cols); // 1?
  ret = clEnqueueNDRangeKernel(app->command_queue, app->multiply_kernel, 1, NULL, &globalSize_multiply, &localSize, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, output_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), output, 0, NULL, NULL);
}

void transpose_and_multiply_matrix_vector_add(const ELEM* A, const ELEM* B, const ELEM* C, const int rows, const int cols) {
  float alpha = 1.0;
  integer incx = 1;
  float beta = 1.0;
  integer incy = 1;
  integer irows = rows;
  integer icols = cols;

  sgemv_("t", &irows, &icols, &alpha, A, &irows, B, &incx, &beta, C, &incy);
}

void negate_matrix(ELEM* input, const int rows, const int cols) {
  int i, j;
  for (i = 0; i < (rows*cols); ++i) {
    input[i] = -1.0 * input[i]; 
  }
}

void cl_negate_matrix(App *app, ELEM* input, cl_mem input_in_cl, const int rows, const int cols) {
  cl_int ret;
  cl_int per_thread = rows;
  size_t localSize = 1;
  size_t globalSize = cols;

  ret = clEnqueueWriteBuffer(app->command_queue, input_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input, 0, NULL, NULL);
  ret = clSetKernelArg(app->negate_kernel, 0, sizeof(cl_mem), (void*)&input_in_cl);
  ret = clSetKernelArg(app->negate_kernel, 1, sizeof(cl_int), (void*)&per_thread);
  ret = clEnqueueNDRangeKernel(app->command_queue, app->negate_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, input_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input, 0, NULL, NULL);
}

void max_of_vector(const ELEM* input, const int num_elems, ELEM *max_value, int *max_pos) {
  if (num_elems >= 1) {
    *max_value = input[0];
    *max_pos = 0;
  }
  int i;
  for (i = 1; i < num_elems; ++i) {
    if (input[i] > *max_value) {
      *max_value = input[i];
      *max_pos = i;
    }
  }
}

void cl_max_of_vector(App *app, const ELEM* input, cl_mem input_in_cl, const int num_elems, ELEM *max_value, int *max_pos) {
  cl_int ret;
  cl_int per_thread = num_elems;
  size_t localSize = 1;
  size_t globalSize = 1;
  
  ret = clEnqueueWriteBuffer(app->command_queue, input_in_cl, CL_TRUE, 0, 1 * sizeof(cl_float), input, 0, NULL, NULL);
  ret = clSetKernelArg(app->max_kernel, 0, sizeof(cl_mem), (void*)&input_in_cl);
  ret = clSetKernelArg(app->max_kernel, 1, sizeof(cl_mem), (void*)&app->cl_max_values);
  ret = clSetKernelArg(app->max_kernel, 2, sizeof(cl_mem), (void*)&app->cl_max_positions);
  ret = clSetKernelArg(app->max_kernel, 3, sizeof(cl_int), (void*)&per_thread);
  ret = clEnqueueNDRangeKernel(app->command_queue, app->max_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, app->cl_max_values, CL_TRUE, 0, 1 * sizeof(cl_float), app->max_values, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, app->cl_max_positions, CL_TRUE, 0, 1 * sizeof(cl_int), app->max_positions, 0, NULL, NULL);
 
  //Normally we would reduce further, but this is a vector
  *max_value = app->max_values[0];
  *max_pos = app->max_positions[0];
}

void pairwise_divide(const ELEM* input1, const ELEM* input2, ELEM* output, const int rows, const int cols) {
  int i, j;
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      output[j * rows + i] = input1[j * rows + i] / input2[j * rows + i];
    }
  }
}

void cl_pairwise_divide(App *app, const ELEM* input1, cl_mem input1_in_cl, const ELEM* input2, cl_mem input2_in_cl, ELEM* output, 
                        cl_mem output_in_cl, const int rows, const int cols) {
  cl_int ret;
  cl_int per_thread = rows;
  size_t localSize = 1;
  size_t globalSize = cols;

  ret = clEnqueueWriteBuffer(app->command_queue, input1_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input1, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(app->command_queue, input2_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), input2, 0, NULL, NULL);
  ret = clSetKernelArg(app->pairwise_kernel, 0, sizeof(cl_mem), (void*)&input1_in_cl);
  ret = clSetKernelArg(app->pairwise_kernel, 1, sizeof(cl_mem), (void*)&input2_in_cl);
  ret = clSetKernelArg(app->pairwise_kernel, 2, sizeof(cl_mem), (void*)&output_in_cl);
  ret = clSetKernelArg(app->pairwise_kernel, 3, sizeof(cl_int), (void*)&per_thread);
  ret = clEnqueueNDRangeKernel(app->command_queue, app->pairwise_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  ret = clEnqueueReadBuffer(app->command_queue, output_in_cl, CL_TRUE, 0, rows * cols * sizeof(cl_float), output, 0, NULL, NULL);
}

void print_matrix(const ELEM* input, const int rows, const int cols) {
  int i, j;
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      printf("%3.3f ", input[j * rows + i]);
    }
    printf("\n");
  }
}

int pivot(App *app) {
  app->curObj = app->newObj;

#ifdef PRINT_MATRICES
  printf("Matrix A:\n");
  print_matrix(app->A, app->NUM_ROWS, app->NUM_COLS);
  printf("Matrix C:\n");
  print_matrix(app->C, 1, app->NUM_COLS);
#endif

  split_matrix(app->A, app->A_B, app->A_N, app->indices_B, app->indices_N, app->NUM_B_INDICES, app->NUM_N_INDICES, app->NUM_ROWS);

#ifdef PRINT_MATRICES
  printf("Matrix A_B:\n");
  print_matrix(app->A_B, app->NUM_B_INDICES, app->NUM_B_INDICES);
  printf("Matrix A_N:\n");
  print_matrix(app->A_N, app->NUM_B_INDICES, app->NUM_N_INDICES);
  printf("\n");
#endif

  split_matrix(app->C, app->C_B, app->C_N, app->indices_B, app->indices_N, app->NUM_B_INDICES, app->NUM_N_INDICES, 1);
#ifdef PRINT_MATRICES
  printf("Matrix cB:\n");
  print_matrix(app->C_B, 1, app->NUM_B_INDICES);
  printf("Matrix cN:\n");
  print_matrix(app->C_N, 1, app->NUM_N_INDICES);
#endif
  
#ifdef PRINT_MATRICES
  printf("Matrix A_B:\n");
  print_matrix(app->A_B, app->NUM_B_INDICES, app->NUM_B_INDICES);
#endif

#ifdef USE_OPENCL
  cl_transpose_matrix(app, app->A_B, app->cl_A_B, app->A_B_trans, app->cl_A_B_trans, app->NUM_B_INDICES, app->NUM_B_INDICES);
#else
  transpose_matrix(app->A_B, app->A_B_trans, app->NUM_B_INDICES, app->NUM_B_INDICES);
#endif

  memcpy(app->C_B_trans, app->C_B, sizeof(ELEM) * app->NUM_B_INDICES);

#ifdef PRINT_MATRICES
  printf("Matrix A_B':\n");
  print_matrix(app->A_B_trans, app->NUM_B_INDICES, app->NUM_B_INDICES);
#endif

#ifdef USE_OPENCL
  cl_solve_matrix(app, app->A_B_trans, app->cl_A_B_trans, app->cl_A_B_trans_tmp, app->C_B_trans, app->cl_C_B_trans, app->C_B_trans, app->cl_C_B_trans, app->NUM_B_INDICES, 1);
#else
  solve_matrix(app->A_B_trans, app->C_B_trans, app->NUM_B_INDICES, 1);
#endif

  app->Y_B = app->C_B_trans;

#ifdef PRINT_MATRICES
  printf("Matrix Y_B:\n");
  print_matrix(app->Y_B, app->NUM_B_INDICES, 1);
#endif

#ifdef USE_OPENCL
  cl_negate_matrix(app, app->C_N, app->cl_C_N, 1, app->NUM_N_INDICES);
#else
  negate_matrix(app->C_N, 1, app->NUM_N_INDICES);
#endif

  transpose_and_multiply_matrix_vector_add(app->A_N, app->Y_B, app->C_N, app->NUM_B_INDICES, app->NUM_N_INDICES);

#ifdef USE_OPENCL
  cl_negate_matrix(app, app->C_N, app->cl_C_N, 1, app->NUM_N_INDICES);
#else
  negate_matrix(app->C_N, 1, app->NUM_N_INDICES);
#endif

  app->zRow = app->C_N;
#ifdef PRINT_MATRICES
  printf("zRow:\n");
  print_matrix(app->zRow, 1, app->NUM_N_INDICES);
#endif

  ELEM max_value;
  int max_pos;

#ifdef USE_OPENCL
  cl_max_of_vector(app, app->zRow, app->cl_C_N, app->NUM_N_INDICES, &max_value, &max_pos);
#else
  max_of_vector(app->zRow, app->NUM_N_INDICES, &max_value, &max_pos);
#endif

  if (max_value < 0) {
    printf("Dictionary is final: %f\n", app->curObj);
    return COMPLETE;
  }

#ifdef PRINT_MATRICES
  printf("Entering variable: %i\n", app->indices_N[max_pos]);
#endif

  int i;
  for (i = 0; i < app->NUM_ROWS; ++i) {
    app->Ad[i] = app->A[app->indices_N[max_pos] * app->NUM_ROWS + i];
  }

#ifdef PRINT_MATRICES
  printf("Ad:\n");
  print_matrix(app->Ad, app->NUM_ROWS, 1);
#endif

#ifdef USE_OPENCL
  cl_solve_matrix(app, app->A_B, app->cl_A_B, app->cl_A_B_tmp, app->Ad, app->cl_Ad, app->Ad, app->cl_Ad, app->NUM_ROWS, 1);
#else
  solve_matrix(app->A_B, app->Ad, app->NUM_ROWS, 1);
#endif

  ELEM *d = app->Ad;

#ifdef PRINT_MATRICES
  printf("d:\n");
  print_matrix(d, app->NUM_ROWS, 1);
#endif
 
#ifdef USE_OPENCL
  cl_pairwise_divide(app, app->B, app->cl_B, d, app->cl_Ad, app->tVec, app->cl_tVec, app->NUM_ROWS, 1);
#else
  pairwise_divide(app->B, d, app->tVec, app->NUM_ROWS, 1);
#endif

#ifdef PRINT_MATRICES
  printf("tVec:\n");
  print_matrix(app->tVec, app->NUM_ROWS, 1);
#endif

  ELEM t = INFINITY;
  int leaveInd = -1;

  for (i = 0; i < app->NUM_ROWS; ++i) {
    if ((app->tVec[i] > 0) && (app->tVec[i] < t)) {
      t = app->tVec[i];
      app->newObj = app->curObj + t * max_value;
      leaveInd = i;
    }
  }

  if (leaveInd < 0) {
    printf("No leaving variable.  Problem is unbounded\n");
    app->newObj = INFINITY;
    return UNBOUNDED;
  }

  int leaveVar = app->indices_B[leaveInd];
#ifdef PRINT_MATRICES
  printf("Leaving var (index: %i): %i\n", leaveInd, leaveVar);
  printf("t: %f\n", t);
#endif
  for (i = 0; i < app->NUM_ROWS; ++i) {
    app->s1[i] = app->B[i] - t * d[i];
  }

#ifdef PRINT_MATRICES
  printf("s1:\n");
  print_matrix(app->s1, app->NUM_ROWS, 1);
#endif

  app->s1[leaveInd] = t;
  app->indices_B[leaveInd] = app->indices_N[max_pos];
  app->indices_N[max_pos] = leaveVar;

#ifdef PRINT_MATRICES
  printf("s1:\n");
  print_matrix(app->s1, app->NUM_ROWS, 1);
#endif

#ifdef PRINT_MATRICES
  printf("B indices:\n");
  for (i = 0; i < app->NUM_B_INDICES; ++i) 
    printf("%i ", app->indices_B[i]);
  printf("\n");

  printf("N indices:\n");
  for (i = 0; i < app->NUM_N_INDICES; ++i) 
    printf("%i ", app->indices_N[i]);
  printf("\n");
#endif

  return INCOMPLETE;
}  

void setup_opencl(App *app) {
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("kernels.cl", "r");
  if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  // Create an OpenCL context
  app->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  app->command_queue = clCreateCommandQueue(app->context, device_id, 0, &ret);

  cl_program program = clCreateProgramWithSource(app->context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

  switch (ret) {
    case (CL_INVALID_CONTEXT) : printf("Invalid context\n"); exit(-1); break;
    case (CL_INVALID_VALUE) : printf("Invalid value\n"); exit(-1); break;
    case (CL_OUT_OF_HOST_MEMORY) : printf("Out of host memory\n"); exit(-1); break;
    //case (CL_SUCCESS) : printf("Program created\n"); break;
  }

  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  app->negate_kernel = clCreateKernel(program, "negate_matrix", &ret);
  app->inverse_kernel = clCreateKernel(program, "inverse", &ret);
  app->multiply_kernel = clCreateKernel(program, "multiply", &ret);
  app->transpose_kernel = clCreateKernel(program, "transpose_matrix", &ret);
  app->pairwise_kernel = clCreateKernel(program, "pairwise_divide_matrix", &ret);
  app->max_kernel = clCreateKernel(program, "max_matrix", &ret);
}

void load_data_file(App *app, const char *filename) {
  FILE *in;
  int i, j;
  cl_int ret;

  if (in = fopen(filename, "rt")) {
    fscanf(in, "%u,%u\n", &app->NUM_ROWS, &app->NUM_COLS);

    app->NUM_ROWS *= 2;  // To allow for both bounds
    app->NUM_COLS += app->NUM_ROWS;

    // save typing
    int NUM_ROWS = app->NUM_ROWS;
    int NUM_COLS = app->NUM_COLS;    

    app->NUM_B_INDICES = app->NUM_ROWS;
    app->NUM_N_INDICES = app->NUM_COLS - app->NUM_ROWS;

    app->A = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_COLS);
    app->C = (ELEM *)malloc(sizeof(ELEM) * NUM_COLS);

    app->B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS);
    app->cl_B = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * sizeof(cl_float), NULL, &ret);

    app->A_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
    app->cl_A_B = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * NUM_ROWS * sizeof(cl_float), NULL, &ret);
    app->cl_A_B_tmp = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * NUM_ROWS * sizeof(cl_float), NULL, &ret);

    app->A_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS) * NUM_ROWS);
    app->C_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS);
    app->cl_C_B = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * sizeof(cl_float), NULL, &ret);
    app->C_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));
    app->cl_C_N = clCreateBuffer(app->context, CL_MEM_READ_WRITE, (NUM_COLS - NUM_ROWS) * sizeof(cl_float), NULL, &ret);
  
    app->A_B_trans = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
    app->cl_A_B_trans = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * NUM_ROWS * sizeof(cl_float), NULL, &ret);
    app->cl_A_B_trans_tmp = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * NUM_ROWS * sizeof(cl_float), NULL, &ret);

    //FIXME: this is a vector and doesn't need a special transpose
    app->C_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));
    app->cl_C_B_trans = clCreateBuffer(app->context, CL_MEM_READ_WRITE, (NUM_COLS - NUM_ROWS) * sizeof(cl_float), NULL, &ret);
  
    app->Y_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

    app->Ad = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
    app->cl_Ad = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * sizeof(cl_float), NULL, &ret);

    app->tVec = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
    app->cl_tVec = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_ROWS * sizeof(cl_float), NULL, &ret);

    app->s1 = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

    app->indices_N = (int *)malloc(sizeof(int) * (app->NUM_N_INDICES));
    app->indices_B = (int *)malloc(sizeof(int) * (app->NUM_B_INDICES));

    app->max_values = (ELEM *)malloc(sizeof(ELEM) * NUM_COLS);
    app->cl_max_values = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_COLS * sizeof(cl_float), NULL, &ret);

    app->max_positions = (int *)malloc(sizeof(int) * NUM_COLS);
    app->cl_max_positions = clCreateBuffer(app->context, CL_MEM_READ_WRITE, NUM_COLS * sizeof(cl_int), NULL, &ret);

    for (i = 0; i < app->NUM_N_INDICES; ++i) {
      app->indices_N[i] = i;
    }

    for (i = 0; i < app->NUM_B_INDICES; ++i) {
      app->indices_B[i] = i + app->NUM_N_INDICES;
    }

    app->curObj = 0;
    app->newObj = 0;
    
    // Read in the objective
    for (i = 0; i < (app->NUM_COLS - app->NUM_ROWS); ++i) {
      if (i)
        fscanf(in, ", %f", &(app->C[i]));
      else
        fscanf(in, "%f", &(app->C[i]));
    }
    fscanf(in, "\n");

    for (i = (app->NUM_COLS - app->NUM_ROWS); i < app->NUM_COLS; ++i) {
      app->C[i] = 0.0;
    }

    // Read in the matrix
    int row_index, col_index;
    for (row_index = 0; row_index < app->NUM_ROWS; row_index+=2) {
      for (col_index = 0; col_index < (app->NUM_COLS - app->NUM_ROWS); ++col_index) {
        float value;
        fscanf(in, col_index ? ", %f" : "%f", &value);
        app->A[col_index * app->NUM_ROWS + row_index] = value;
        app->A[col_index * app->NUM_ROWS + row_index + 1] = -value;
      }
    }
    fscanf(in, "\n");

    // Fill out the identity side of A
    for (i = 0; i < app->NUM_ROWS; ++i) {
      for (j = 0; j < app->NUM_ROWS; ++j) {
        if (i == j)
          app->A[(app->NUM_COLS - app->NUM_ROWS) * app->NUM_ROWS + i * app->NUM_ROWS + j] = 1; 
        else
          app->A[(app->NUM_COLS - app->NUM_ROWS) * app->NUM_ROWS + i * app->NUM_ROWS + j] = 0; 
      }
    }

    // Read the constraint bounds
    float item;
    for (i = 0; i < app->NUM_ROWS / 2; ++i) {
      if (i)
        fscanf(in, ", %f", &item);
      else
        fscanf(in, "%f", &item);
      
      app->B[i * 2 + 1] = item;
    }
    fscanf(in, "\n");

    // Read the constraint bounds
    for (i = 0; i < app->NUM_ROWS / 2; ++i) {
      if (i)
        fscanf(in, ", %f", &item);
      else
        fscanf(in, "%f", &item);
      
      app->B[i * 2] = item;
    }
    fscanf(in, "\n");

    fclose(in);
  }
  else {
    fprintf(stderr, "Could not open %s\n", filename);
    exit(-1);
  }  
}

int main(int argc, char *argv[]) {
  App app;

  if (argc > 1) {
    printf("Loading: %s\n", argv[1]);
    setup_opencl(&app);
    load_data_file(&app, argv[1]);
    printf("Pivoting...\n");
    while (pivot(&app) == INCOMPLETE);
    //pivot(&app);
  }
  else {
    printf("Specify input file\n");
  }
   
}

