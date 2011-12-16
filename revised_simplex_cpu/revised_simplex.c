#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>
#include <math.h>

#define ELEM float 

#include "f2c.h"

#define INCOMPLETE 1
#define COMPLETE 0
#define ERROR -1
#define UNBOUNDED -2

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

void invert_and_multiply_matrix_vector_minus(const ELEM* A, const ELEM* B, const ELEM* C, const int rows, const int cols) {
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

void pairwise_divide(const ELEM* input1, const ELEM* input2, ELEM* output, const int rows, const int cols) {
  int i, j;
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      output[j * rows + i] = input1[j * rows + i] / input2[j * rows + i];
    }
  }
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

typedef struct App {
  //Foundational parts
  ELEM *A;   // Our main data matrix
  ELEM *A_B; // The current basis of the data matrix
  ELEM *A_N; // The current independent section of the data matrix
  ELEM *B;   // The current row coefficients
  ELEM *C;   // Objective column
  ELEM *C_B; // The basis side of the objective function
  ELEM *C_N; // The independent side of the objective function

  int NUM_ROWS;   // The total number of rows of A
  int NUM_COLS;   // The total number of cols of A
  int *indices_B; // Indices of the current basis columns
  int *indices_N; // Indices of the current independent columns

  int NUM_B_INDICES; // Helpful quick lookup for number of basic variables
  int NUM_N_INDICES; // Helpful quick lokoup for number of independent variables

  //Helpful temporaries
  ELEM *A_B_trans; // The transposed A_B
  ELEM *C_B_trans; // The transposed C_B
  ELEM *Y_B;       // The intermediate solve of A_B' \ cB'
  ELEM *Y_B_trans; // The transposed Y_B
  ELEM *zRow;      // Temporary calculation of the current objective
  ELEM *Ad;        // Entering variable temporary
  ELEM *tVec;      // Leaving variable temporary
  ELEM *s1;        // The new coefficients after pivot  

  ELEM curObj;  // The current objective value
  ELEM newObj;  // The new objective value at the end of this pivot
  
} App;

#define PRINT_MATRICES

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
  transpose_matrix(app->A_B, app->A_B_trans, app->NUM_B_INDICES, app->NUM_B_INDICES);
  transpose_matrix(app->C_B, app->C_B_trans, 1, app->NUM_B_INDICES);
#ifdef PRINT_MATRICES
  printf("Matrix A_B':\n");
  print_matrix(app->A_B_trans, app->NUM_B_INDICES, app->NUM_B_INDICES);
  printf("Matrix C_B':\n");
  print_matrix(app->C_B_trans, app->NUM_B_INDICES, 1);
#endif

  solve_matrix(app->A_B_trans, app->C_B_trans, app->NUM_B_INDICES, 1);
  app->Y_B = app->C_B_trans;

#ifdef PRINT_MATRICES
  printf("Matrix Y_B:\n");
  print_matrix(app->Y_B, app->NUM_B_INDICES, 1);
#endif

  negate_matrix(app->C_N, 1, app->NUM_N_INDICES);
  invert_and_multiply_matrix_vector_minus(app->A_N, app->Y_B, app->C_N, app->NUM_B_INDICES, app->NUM_N_INDICES);
  negate_matrix(app->C_N, 1, app->NUM_N_INDICES);
  app->zRow = app->C_N;
#ifdef PRINT_MATRICES
  printf("zRow:\n");
  print_matrix(app->zRow, 1, app->NUM_N_INDICES);
#endif

  ELEM max_value;
  int max_pos;

  max_of_vector(app->zRow, app->NUM_N_INDICES, &max_value, &max_pos);

  if (max_value < 0) {
    printf("Dictionary is final: %f\n", app->curObj);
    return COMPLETE;
  }

  printf("Entering variable: %i\n", app->indices_N[max_pos]);
  int i;
  for (i = 0; i < app->NUM_ROWS; ++i) {
    app->Ad[i] = app->A[app->indices_N[max_pos] * app->NUM_ROWS + i];
  }

#ifdef PRINT_MATRICES
  printf("Ad:\n");
  print_matrix(app->Ad, app->NUM_ROWS, 1);
#endif

  solve_matrix(app->A_B, app->Ad, app->NUM_ROWS, 1);

  ELEM *d = app->Ad;

#ifdef PRINT_MATRICES
  printf("d:\n");
  print_matrix(d, app->NUM_ROWS, 1);
#endif
 
  pairwise_divide(app->B, d, app->tVec, app->NUM_ROWS, 1);
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

  printf("B indices:\n");
  for (i = 0; i < app->NUM_B_INDICES; ++i) 
    printf("%i ", app->indices_B[i]);
  printf("\n");

  printf("N indices:\n");
  for (i = 0; i < app->NUM_N_INDICES; ++i) 
    printf("%i ", app->indices_N[i]);
  printf("\n");

  return INCOMPLETE;
}  

void load_data_file(App *app, const char *filename) {
  FILE *in;
  int i, j;

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
    app->A_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
    app->A_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS) * NUM_ROWS);
    app->C_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS);
    app->C_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));
  
    app->A_B_trans = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
    //FIXME: this is a vector and doesn't need a special transpose
    app->C_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));
  
    app->Y_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

    app->Ad = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
    app->tVec = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
    app->s1 = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

    app->indices_N = (int *)malloc(sizeof(int) * (app->NUM_N_INDICES));
    app->indices_B = (int *)malloc(sizeof(int) * (app->NUM_B_INDICES));

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
    load_data_file(&app, argv[1]);
    while (pivot(&app) == INCOMPLETE);
    //pivot(&app);
  }
  else {
    printf("Specify input file\n");
  }
   
    
/*
  //int NUM_ROWS = 3;
  //int NUM_COLS = 6;
  int NUM_ROWS = 3;
  int NUM_COLS = 5;

  app.NUM_ROWS = NUM_ROWS;
  app.NUM_COLS = NUM_COLS;
  app.NUM_B_INDICES = app.NUM_ROWS;
  app.NUM_N_INDICES = app.NUM_COLS - app.NUM_ROWS;

  app.A = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_COLS);
  app.C = (ELEM *)malloc(sizeof(ELEM) * NUM_COLS);
  app.B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS);
  app.A_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
  app.A_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS) * NUM_ROWS);
  app.C_B = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS);
  app.C_N = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));
  
  app.A_B_trans = (ELEM *)malloc(sizeof(ELEM) * NUM_ROWS * NUM_ROWS);
  //FIXME: this is a vector and doesn't need a special transpose
  app.C_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_COLS - NUM_ROWS));

  app.Y_B_trans = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

  app.Ad = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
  app.tVec = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));
  app.s1 = (ELEM *)malloc(sizeof(ELEM) * (NUM_ROWS));

  app.indices_N = (int *)malloc(sizeof(int) * (NUM_ROWS));
  app.indices_B = (int *)malloc(sizeof(int) * (NUM_COLS - NUM_ROWS));

  app.curObj = 0;
  app.newObj = 0;
*/
/*
  ELEM A[NUM_ROWS * NUM_COLS] = {2, 3, 1, 2, -2, -3, -1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  ELEM C[NUM_COLS] = {1, 3, -1, 0, 0, 0};
  ELEM B[NUM_ROWS] = {10, 10, 10};
  ELEM A_B[NUM_B_INDICES * NUM_B_INDICES] = {0};
  ELEM A_N[NUM_N_INDICES * NUM_B_INDICES] = {0};
  ELEM C_B[NUM_B_INDICES] = {0};
  ELEM C_N[NUM_N_INDICES] = {0};

  ELEM A_B_trans[NUM_B_INDICES * NUM_B_INDICES] = {0};
  ELEM C_B_trans[NUM_B_INDICES] = {0};

  ELEM* Y_B;
  ELEM Y_B_trans[NUM_B_INDICES] = {0};
  ELEM* zRow;

  //FIXME
  ELEM curObj = 0;
  ELEM newObj;

  int indices_B[NUM_B_INDICES] = {3, 4, 5};
  int indices_N[NUM_N_INDICES] = {0, 1, 2};
*/
/* 
  //UNBOUNDED
  app.A[0] = 2;
  app.A[1] = 3;
  app.A[2] = 1;
  app.A[3] = 2;
  app.A[4] = -2;
  app.A[5] = -3;
  app.A[6] = -1;
  app.A[7] = 1;
  app.A[8] = 1;
  app.A[9] = 1;
  app.A[10] = 0;
  app.A[11] = 0;
  app.A[12] = 0;
  app.A[13] = 1;
  app.A[14] = 0;
  app.A[15] = 0;
  app.A[16] = 0;
  app.A[17] = 1;

  app.B[0] = 10;
  app.B[1] = 10;
  app.B[2] = 10;

  app.C[0] = 1;
  app.C[1] = 3;
  app.C[2] = -1;
  app.C[3] = 0;
  app.C[4] = 0;
  app.C[5] = 0;
  
  app.indices_B[0] = 3;
  app.indices_B[1] = 4;
  app.indices_B[2] = 5;

  app.indices_N[0] = 0;
  app.indices_N[1] = 1;
  app.indices_N[2] = 2;
*/
/*
  //BOUNDED
  app.A[0] = -4;
  app.A[1] = -2;
  app.A[2] = 1;
  app.A[3] = -1;
  app.A[4] = 1;
  app.A[5] = 2;
  app.A[6] = 1;
  app.A[7] = 0;
  app.A[8] = 0;
  app.A[9] = 0;
  app.A[10] = 1;
  app.A[11] = 0;
  app.A[12] = 0;
  app.A[13] = 0;
  app.A[14] = 1;

  app.B[0] = 4;
  app.B[1] = 8;
  app.B[2] = 4;
  
  app.C[0] = -3;
  app.C[1] = 5;
  app.C[2] = 0;
  app.C[3] = 0;
  app.C[4] = 0;

  app.indices_B[0] = 2;
  app.indices_B[1] = 3;
  app.indices_B[2] = 4;

  app.indices_N[0] = 0;
  app.indices_N[1] = 1;
*/
}

