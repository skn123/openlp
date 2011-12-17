__kernel void subtract_transpose_multiply(__global float *a, __global float *b, __global float *y, int arows){
	int temp = get_global_id(0);

        float sum = 0.0;
        for(int i = 0; i < arows; i++){
                sum += a[temp * arows + i] * b[i];
        }
        y[temp] = y[temp] - sum;
}


__kernel void multiply(__global float *a, __global float *b, __global float *x, int acols, int arows, int bcols, int brows){
	int temp = get_global_id(0);

	int result = (int)temp/(int)arows;

	float sum = 0.0;
	for(int i = 0; i < acols; i++){
		sum += a[temp%arows + i*arows]*b[i + (result*brows)];
	}
	x[temp] = sum;
}

/*

__kernel void multiply(__global float *a, __global float *b, __global float *x, int arows, int brows, int bcols) {
	// working C++ code for multiplying two matrices that are in 1d column-major order 

    for(int i = 0; i < bcols; i++){
      for(int j = 0; j < arows; j++){
        x[i*arows+j] = 0;
        for(int k = 0; k < brows; k++){
          x[i*arows+j] += a[j+k*arows]*b[i*brows+k];
        }
      }
    }
}
*/

__kernel void pairwise_divide_matrix(__global float *in1, __global float *in2, __global float *out, int size) {
  int id = get_global_id(0);

  for (int i = 0; i < size; ++i) {
    out[i + id * size] = in1[i + id * size] / in2[i + id * size];
  }

}

__kernel void transpose_matrix(__global float *in, __global float *out, int size) {
  int id = get_global_id(0);

  for (int j = 0; j < size; ++j) {
    out[id + j * size] = in[j + id * size];
  }

}

__kernel void negate_matrix(__global float *inout, int size) {
  int id = get_global_id(0);
  for (int j = 0; j < size; ++j) {
    inout[j + id * size] = inout[j + id * size] * -1;
  }
}

__kernel void max_matrix(__global float *in, __global float *out, __global int *out_pos, int size) {
  int id = get_global_id(0);

  float max_val = -1000000;
  int max_pos = -1;

  for (int j = 0; j < size; ++j) {
    if (max_val < in[id * size + j]) {
      max_pos = id * size + j; 
      max_val = in[max_pos];
    }
  }
  out[id] = max_val;
  out_pos[id] = max_pos; 
}

//Takes in row-major matrix and produces the col-major inverse of the matrix
__kernel void inverse(__global float *Min, __global float *Mout, int actualsize) {
    /* Loop variables */
    int i, j, k;
    /* Sum variables */
    float sum,x;
    
    /*  Copy the input matrix to output matrix */
    for(i=0; i<actualsize*actualsize; i++) { Mout[i]=Min[i]; }
    
    /* Add small value to diagonal if diagonal is zero */
    for(i=0; i<actualsize; i++)
    { 
        j=i*actualsize+i;
        if((Mout[j]<1e-12)&&(Mout[j]>-1e-12)){ Mout[j]=1e-12; }
    }
    
    /* Matrix size must be larger than one */
    if (actualsize <= 1) return;
    
    for (i=1; i < actualsize; i++) {
        Mout[i] /= Mout[0]; /* normalize row 0 */
    }
    
    for (i=1; i < actualsize; i++)  {
        for (j=i; j < actualsize; j++)  { /* do a column of L */
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += Mout[j*actualsize+k] * Mout[k*actualsize+i];
            }
            Mout[j*actualsize+i] -= sum;
        }
        if (i == actualsize-1) continue;
        for (j=i+1; j < actualsize; j++)  {  /* do a row of U */
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += Mout[i*actualsize+k]*Mout[k*actualsize+j];
            }
            Mout[i*actualsize+j] = (Mout[i*actualsize+j]-sum) / Mout[i*actualsize+i];
        }
    }
    for ( i = 0; i < actualsize; i++ )  /* invert L */ {
        for ( j = i; j < actualsize; j++ )  {
            x = 1.0;
            if ( i != j ) {
                x = 0.0;
                for ( k = i; k < j; k++ ) {
                    x -= Mout[j*actualsize+k]*Mout[k*actualsize+i];
                }
            }
            Mout[j*actualsize+i] = x / Mout[j*actualsize+j];
        }
    }
    for ( i = 0; i < actualsize; i++ ) /* invert U */ {
        for ( j = i; j < actualsize; j++ )  {
            if ( i == j ) continue;
            sum = 0.0;
            for ( k = i; k < j; k++ ) {
                sum += Mout[k*actualsize+j]*( (i==k) ? 1.0 : Mout[i*actualsize+k] );
            }
            Mout[i*actualsize+j] = -sum;
        }
    }
    for ( i = 0; i < actualsize; i++ ) /* final inversion */ {
        for ( j = 0; j < actualsize; j++ )  {
            sum = 0.0;
            for ( k = ((i>j)?i:j); k < actualsize; k++ ) {
                sum += ((j==k)?1.0:Mout[j*actualsize+k])*Mout[k*actualsize+i];
            }
            Mout[j*actualsize+i] = sum;
        }
    }

	int count = 0;
	for(i = 0; i < actualsize; i++){
		for(j = 0; j < actualsize*actualsize; j+=actualsize){
			Min[count] = Mout[i+j];
			count++;
		}
	}

	for(i = 0; i < actualsize*actualsize; i++)
		Mout[i] = Min[i];
}

