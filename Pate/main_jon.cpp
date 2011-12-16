#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <limits>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main(){

  double inf = numeric_limits<double>::infinity(); //Adds functionality for infinite bounds
  ifstream infile("input.txt");
  vector <double> v_input;
  string line, token;
  stringstream iss;

  //Gets Number of Constants and Variables from file
  int numCons, numVars;
  getline(infile, line);
  iss << line;
  getline(iss, token, ',');
    numCons = atoi(token.c_str());
      iss.clear();
  getline(iss, token, ',');
    numVars = atoi(token.c_str());
      iss.clear();


  //Skip empty line
  getline(infile, line);

  //Create Z array (objective function)
  double *Z = new double [numVars];
  getline(infile, line);
  iss << line;
  for(int i = 0; i < numVars; i++){
    getline(iss, token, ',');
	Z[i] = atof(token.c_str());
    iss.clear();
  }

  getline(infile, line); //skip empty line

  //Create Main Matrix
  double *matrix = new double [numCons*numVars];
  for(int i = 0; i < numCons; i++){
    getline(infile, line);
    iss << line;
    for(int j = 0; j < numVars; j++){
      getline(iss, token, ',');
      matrix[i * (numVars) + j] = (atof(token.c_str()));
    }
    iss.clear();
  }

  getline(infile, line); //skip empty line

  //create b array (RHS values)
  double *b = new double [numCons];
  getline(infile, line);
  iss << line;
  for(int i = 0; i < numCons; i++){
    getline(iss, token, ',');
    b[i] = atof(token.c_str());
  }
  iss.clear();

  //Create B matrix (identity matrix)
  double *B = new double[numCons*numCons];
  for(int i = 0; i < numCons*numCons; i++){
	B[i] = 0;
  }
  for(int i = 0; i < numCons*numCons; i+=(numCons+1)){
	B[i] = 1;
  }

  //******************************************************************************************

    // Create the two input vectors
    cl_float *X = (cl_float *)malloc(numCons*numVars * sizeof(cl_float));

    for(int i = 0; i < numCons*numVars; i++) {
        ((float*)X)[i] = (float)matrix[i];
    }

    cl_float *Min = (cl_float *)malloc(3*3*sizeof(cl_float));
    Min[0] = 1; Min[1] = 3; Min[2] = 1;
    Min[3] = 1; Min[4] = 6.1; Min[5] = 2;
    Min[6] = 2; Min[7] = 3; Min[8] = 4;

    cl_float *Min2 = (cl_float *)malloc(3*3*sizeof(cl_float));
    Min2[0] = 1; Min2[1] = 6; Min2[2] = 3;
    Min2[3] = 1; Min2[4] = 2; Min2[5] = 6;
    Min2[6] = 2; Min2[7] = 6; Min2[8] = 12;


    // Load the kernel source code into the array source_str
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
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem matrix_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,numCons*numVars * sizeof(cl_float), NULL, &ret);
    cl_mem Min_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_float), NULL, &ret);
    cl_mem Min2_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_float), NULL, &ret);
    cl_mem Mout_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_float), NULL, &ret);
    cl_mem Maxout_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_float), NULL, &ret);
    cl_mem Maxout_pos_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_int), NULL, &ret);

	cl_float amatrix[] = {1, 4, 2, 5, 3, 6};
	cl_float bmatrix[] = {9, 8, 7};
	cl_float xmatrix[2] = {0};
	cl_int arows = 3;
	cl_int acols = 2;
	cl_int brows = 1;
	cl_int bcols = 3;

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, arows*acols*sizeof(cl_float),NULL,&ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, brows*bcols*sizeof(cl_float),NULL,&ret);
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, arows*bcols*sizeof(cl_float),NULL,&ret);
 
    // Copy the lists X and Y to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, matrix_mem_obj, CL_TRUE, 0, numCons*numVars * sizeof(cl_float), X, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, Min_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), Min, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, Min2_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), Min2, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, arows*acols*sizeof(cl_float), amatrix, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, brows*bcols*sizeof(cl_float), bmatrix, 0 ,NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    switch (ret) {
      case (CL_INVALID_CONTEXT) : printf("Invalid context\n"); exit(-1); break;
      case (CL_INVALID_VALUE) : printf("Invalid value\n"); exit(-1); break;
      case (CL_OUT_OF_HOST_MEMORY) : printf("Out of host memory\n"); exit(-1); break;
      case (CL_SUCCESS) : printf("Program created\n"); break; 
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel inverse_kernel = clCreateKernel(program, "inverse", &ret);
    cl_kernel multiply_kernel = clCreateKernel(program, "multiply", &ret);

    cl_kernel negate_kernel = clCreateKernel(program, "negate_matrix", &ret);
    cl_kernel transpose_kernel = clCreateKernel(program, "transpose_matrix", &ret);
    cl_kernel pairwise_kernel = clCreateKernel(program, "pairwise_divide_matrix", &ret);

    cl_kernel max_kernel = clCreateKernel(program, "max_matrix", &ret);

    cl_int actualSize1 = 3;
    ret = clSetKernelArg(negate_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
    ret = clSetKernelArg(negate_kernel, 1, sizeof(cl_int), (void*)&actualSize1);
    
    size_t localSize = 1;
    size_t globalSize = 3;
    //ret = clEnqueueNDRangeKernel(command_queue, negate_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    ret = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
    ret = clSetKernelArg(transpose_kernel, 1, sizeof(cl_mem), (void*)&Mout_mem_obj);
    ret = clSetKernelArg(transpose_kernel, 2, sizeof(cl_int), (void*)&actualSize1);
    //ret = clEnqueueNDRangeKernel(command_queue, transpose_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
 
    // Set the arguments of the inverse kernel
    //__kernel void inverse(__global float *Min, __global float *Mout, int actualsize) {

    ret = clSetKernelArg(inverse_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
    ret = clSetKernelArg(inverse_kernel, 1, sizeof(cl_mem), (void*)&Mout_mem_obj);
    cl_int actualSize = 3;
    ret = clSetKernelArg(inverse_kernel, 2, sizeof(cl_int), (void*)&actualSize);

    ret = clSetKernelArg(pairwise_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
    ret = clSetKernelArg(pairwise_kernel, 1, sizeof(cl_mem), (void*)&Min2_mem_obj);
    ret = clSetKernelArg(pairwise_kernel, 2, sizeof(cl_mem), (void*)&Mout_mem_obj);
    ret = clSetKernelArg(pairwise_kernel, 3, sizeof(cl_int), (void*)&actualSize);
    ret = clEnqueueNDRangeKernel(command_queue, pairwise_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    ret = clSetKernelArg(max_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
    ret = clSetKernelArg(max_kernel, 1, sizeof(cl_mem), (void*)&Maxout_mem_obj);
    ret = clSetKernelArg(max_kernel, 2, sizeof(cl_mem), (void*)&Maxout_pos_mem_obj);
    ret = clSetKernelArg(max_kernel, 3, sizeof(cl_int), (void*)&actualSize);
    ret = clEnqueueNDRangeKernel(command_queue, max_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Set the arguments of the multiply kernel
	//__kernel void multiply(__global float *a, __global float *b, __global float *x, int arows, int brows, int bcols) {
	/*
	ret = clSetKernelArg(multiply_kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 2, sizeof(cl_mem), (void*)&x_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 3, sizeof(cl_int), (void*)&arows);
	ret = clSetKernelArg(multiply_kernel, 4, sizeof(cl_int), (void*)&brows);
	ret = clSetKernelArg(multiply_kernel, 5, sizeof(cl_int), (void*)&bcols);
	*/

	//__kernel void multiply(__global float *a, __global float *b, __global float *x, int acols int arows, int bcols, int brows){
	ret = clSetKernelArg(multiply_kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 2, sizeof(cl_mem), (void*)&x_mem_obj);
	ret = clSetKernelArg(multiply_kernel, 3, sizeof(cl_int), (void*)&acols);
	ret = clSetKernelArg(multiply_kernel, 4, sizeof(cl_int), (void*)&arows);
	ret = clSetKernelArg(multiply_kernel, 5, sizeof(cl_int), (void*)&bcols);
	ret = clSetKernelArg(multiply_kernel, 6, sizeof(cl_int), (void*)&brows);

    // Execute the OpenCL kernel on the list
    //ret = clEnqueueTask(command_queue, inverse_kernel, 0, NULL,NULL);
	//ret = clEnqueueTask(command_queue, multiply_kernel, 0, NULL, NULL);

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 3;
    localWorkSize[1] = 3;
    globalWorkSize[0] = 3;
    globalWorkSize[1] = 3;

	//ret = clEnqueueNDRangeKernel(command_queue, multiply_kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	cl_float readFromInverse[9] = {1.1};
	cl_float readFromReverse[9] = {1.1};
	cl_float readFromTranspose[9] = {1.1};
	cl_float readFromPairwise[9] = {1.1};
	cl_float readFromMax[9] = {1.1};
	cl_int readFromMaxPos[9] = {1};
 
    // Read the memory buffer C on the device to the local variable C
	ret = clEnqueueReadBuffer(command_queue, Min_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), readFromInverse, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, Mout_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), readFromTranspose, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, Mout_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), readFromPairwise, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, Maxout_mem_obj, CL_TRUE, 0, 3*sizeof(cl_float), readFromMax, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, Maxout_pos_mem_obj, CL_TRUE, 0, 3*sizeof(cl_float), readFromMaxPos, 0, NULL, NULL);
	
	ret = clEnqueueReadBuffer(command_queue, x_mem_obj, CL_TRUE, 0, acols*brows*sizeof(cl_float), xmatrix, 0, NULL, NULL);

    // Display the result to the screen
    cout << "In: " << std::endl;
    for(int i = 0; i < 9; ++i) 
      cout << Min[i] << " ";
    cout << std::endl;

    cout << "Transpose: " << std::endl;
    for(int i = 0; i < 9; i++)
      cout << readFromTranspose[i] << " ";
    cout << endl;
 
    cout << "Reverse: " << std::endl;
    for(int i = 0; i < 9; i++)
      cout << readFromReverse[i] << " ";
    cout << endl;
 
    cout << "Inverse: " << std::endl;
    for(int i = 0; i < 9; i++)
      cout << readFromInverse[i] << " ";
    cout << endl;
 
    cout << "Pairwise: " << std::endl;
    for(int i = 0; i < 9; i++)
      cout << readFromPairwise[i] << " ";
    cout << endl;
    
    cout << "Max: " << std::endl;
    for(int i = 0; i < 9; i++)
      cout << readFromMaxPos[i] << ": " << readFromMax[i] << " " << std::endl;
    cout << endl;
    
    for(int i = 0; i < 2; i++)
      cout << xmatrix[i] << " ";
    cout << endl;

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(inverse_kernel);
	ret = clReleaseKernel(multiply_kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(matrix_mem_obj);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(x_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(X);
  
  cout << "Program is now finished...\n";
  cin.get();
  return 0;
}
