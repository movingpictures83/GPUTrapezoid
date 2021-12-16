/* Title: Trapezoidal Sums
   This program uses trapezoidal sum integration with varying numbers of trapezoids
   in order to calculate the integral of a curve and chooses the smallest number of
   trapezoids needed to produce a result within a preset epsilon range. 

   Group Members: Juan Nunuez, Rafael McCormack, Bhavyta , Nicolas Dabdoub

*/ 

#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include "GPUTrapezoidPlugin.h"


void GPUTrapezoidPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 nTrapsPow2 = atoi(parameters["nTrapsPow2"].c_str());
 nSumsPow2 = atoi(parameters["nSumsPow2"].c_str());
 x1 = atoi(parameters["x1"].c_str());
 x2 = atoi(parameters["x2"].c_str());
 x2 = atoi(parameters["x2"].c_str());
 answer = atof(parameters["answer"].c_str());
 epsilon = atof(parameters["epsilon"].c_str());

}

void GPUTrapezoidPlugin::run() {
	int S = 0;
	int i, x = 0;
 	double* areaPow2;
	double* sumPow2;
	double* answersPow2;
	double* areaLinear;
	double* sumLinear;
	int numThreads = 1024;
	int numCores = nTrapsPow2 / numThreads + 1;
	double* answersLinear;

	//allocates memory in the cpu for the sums of trap sets increasing by powers of 2
	//allocates memory in the gpu for sum and area array for trap sets increasing by powers of 2
	answersPow2 = (double*)malloc(nSumsPow2*sizeof(double));
	cudaMalloc(&areaPow2, nTrapsPow2*sizeof(double));
	cudaMalloc(&sumPow2, nSumsPow2*sizeof(double));

	//device call
	trapAreaPow2<<<numCores, numThreads>>>(x1, x2, areaPow2, nTrapsPow2);
		
	numCores = nSumsPow2 / numThreads + 1;
	trapSumPow2<<<numCores, numThreads>>>(sumPow2, nSumsPow2, areaPow2);
	
	cudaMemcpy(answersPow2, sumPow2, nSumsPow2*sizeof(double), cudaMemcpyDeviceToHost); 
	
	//determines and records which sums have been within the set epsilon range of the actual predetermined answer
	//this loop is only concerned with sets of trapezoids increasing by powers of 2
	//this loop marks x as the index of the smallest sum that met epsilon.
	for(i = nSumsPow2 - 1; i >= 0; i--)
	{
		//printf("answersPow2[%d] = %f\n", i, answersPow2[i]);
		if((answer - answersPow2[i]) >= -epsilon && (answer - answersPow2[i]) <= epsilon)
		{
			x = i + 1;
		}
	}
	S = exp2((double) x) / 2;
	//printf("x = %d and S = %d\n", x, S);
	//printf("S = %d, the sum with %d trapazoids was %f\n", S, S*2, answersPow2[x-1]);
	
	//focuses on the linearly increasing range of trapezoids between the 2^S and the 2^(S+1) 
	//determines the smallest number of trapezoids that has a sum within the epsilon range
	if(S == 1)
		printf("This is the minimum number of trapezoids needed to compute a trapezoidal sum that is within our epsilon of the actual answer: %d\n", S*2);
	else if(S == 0)
		printf("1024 trapazoids is too few to get an answer within epsilon.\n");
	else
	{
		nTraps = ((S + 1) * S) + (((S * S) + S) / 2);
		nSums = S + 1;
		cudaMalloc(&sumLinear, nSums * sizeof(double));
		cudaMalloc(&areaLinear, nTraps * sizeof(double));
		numCores = (nTraps/numThreads) + 1;
		trapArea<<<numCores, numThreads>>>(x1, x2, S, areaLinear, nTraps);
		numCores = (nSums/numThreads) + 1;
		trapSum<<<numCores, numThreads>>>(S, sumLinear, nSums, areaLinear);
		answersLinear = (double*) malloc(nSums * sizeof(double));
		cudaMemcpy(answersLinear, sumLinear, nSums*sizeof(double), cudaMemcpyDeviceToHost);
		for(i=nSums-1; i >= 0; i--)
		{
			//printf("answersLinear[%d] = %f\n", i, answersLinear[i]);
			if((answer - answersLinear[i]) >= -epsilon && (answer - answersLinear[i]) <= epsilon)
				x = i;
		}
		x = S + x;
		printf("This is the minimum number of trapezoids needed to compute a trapezoidal sum that is within our epsilon of the actual answer: %d\n", x);
		cudaFree(&sumLinear);
		cudaFree(&areaLinear);
		free(answersLinear);
	}
	cudaFree(&areaPow2);
	cudaFree(&sumPow2);
	free(answersPow2);

}

void GPUTrapezoidPlugin::output(std::string file) {}


PluginProxy<GPUTrapezoidPlugin> GPUTrapezoidPluginProxy = PluginProxy<GPUTrapezoidPlugin>("GPUTrapezoid", PluginManager::getInstance());

