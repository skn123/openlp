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
#include "matrix.h"

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

	//Create the Identity matrix for testing
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
 
    // Copy the lists X and Y to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, matrix_mem_obj, CL_TRUE, 0, numCons*numVars * sizeof(cl_float), X, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, Min_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), Min, 0, NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "inverse", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Min_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Mout_mem_obj);
	cl_int actualSize = 3;
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&actualSize);
	
	/*
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_mem_obj);
	cl_int matrixSize = numCons;
	ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&matrixSize);
	*/


    // Execute the OpenCL kernel on the list
    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);

	cl_float meh[9] = {1.1};
 
    // Read the memory buffer C on the device to the local variable C
   // ret = clEnqueueReadBuffer(command_queue, matrix_mem_obj, CL_TRUE, 0,numCons*numVars * sizeof(cl_float), X, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, Mout_mem_obj, CL_TRUE, 0, 3*3*sizeof(cl_float), meh, 0, NULL, NULL);
 
    // Display the result to the screen
    for(int i = 0; i < 9; i++)
		cout << meh[i] << " ";
	cout << endl;
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(matrix_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(X);
  
  cout << "Program is now finished...\n";
  cin.get();
  return 0;
}