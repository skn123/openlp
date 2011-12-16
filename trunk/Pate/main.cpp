#include <iostream>
#include <string>
#include <stdio.h>
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <limits>
#include <iomanip>

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
	Min[3] = 1; Min[4] = 1; Min[5] = 2;
	Min[6] = 2; Min[7] = 3; Min[8] = 4;

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
	cl_mem Mout_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(cl_float), NULL, &ret);


	cl_int arows = 3;
	cl_int acols = 3;
	cl_int brows = 3;
	cl_int bcols = 3;
	cl_float amatrix[] = {1, 4, 2, 5, 3, 6, 6, 8, 3};
	cl_float bmatrix[] = {9, 8, 7, 1, 2, 3, 4, 5, 6};
	cl_float *xmatrix = new cl_float[arows*bcols];
	

	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, arows*acols*sizeof(cl_float),NULL,&ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, brows*bcols*sizeof(cl_float),NULL,&ret);
	cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, arows*bcols*sizeof(cl_float),NULL,&ret);
 
    // Copy the lists X and Y to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, matrix_mem_obj, CL_TRUE, 0, numCons*numVars * sizeof(cl_float), X, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, Min_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), Min, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, arows*acols*sizeof(cl_float), amatrix, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, brows*bcols*sizeof(cl_float), bmatrix, 0 ,NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel inverse_kernel = clCreateKernel(program, "inverse", &ret);
	cl_kernel multiply_kernel = clCreateKernel(program, "multiply", &ret);
 
    // Set the arguments of the inverse kernel
	//__kernel void inverse(__global float *Min, __global float *Mout, int actualsize) {
    ret = clSetKernelArg(inverse_kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
	ret = clSetKernelArg(inverse_kernel, 1, sizeof(cl_mem), (void*)&Mout_mem_obj);
	cl_int actualSize = 3;
	ret = clSetKernelArg(inverse_kernel, 2, sizeof(cl_int), (void*)&actualSize);

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
    ret = clEnqueueTask(command_queue, inverse_kernel, 0, NULL,NULL);
	//ret = clEnqueueTask(command_queue, multiply_kernel, 0, NULL, NULL);

	size_t localWorkSize, globalWorkSize;
	localWorkSize = 1;
    globalWorkSize = arows*bcols;

	ret = clEnqueueNDRangeKernel(command_queue, multiply_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

	cl_float readFromInverse[9] = {1.1};
 
    // Read the memory buffer C on the device to the local variable C
	ret = clEnqueueReadBuffer(command_queue, Mout_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), readFromInverse, 0, NULL, NULL);
	
	ret = clEnqueueReadBuffer(command_queue, x_mem_obj, CL_TRUE, 0, arows*bcols*sizeof(cl_float), xmatrix, 0, NULL, NULL);

    // Display the result to the screen
    for(int i = 0; i < 9; i++)
		cout << readFromInverse[i] << " ";
	cout << endl;
 
	for(int i = 0; i < arows*bcols; i++)
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