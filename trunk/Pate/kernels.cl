__kernel void invert(__global const float *A, __global float *C, int matrixSize) {
	for(int i = 0; i < matrixSize;i++){
		C[i] = A[i];
	}
}