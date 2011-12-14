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