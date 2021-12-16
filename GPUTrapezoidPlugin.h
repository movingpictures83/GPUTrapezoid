#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

#include "function.h"

class GPUTrapezoidPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
int nTrapsPow2;
int nSumsPow2;
int nTraps;
int nSums;
int x1;
int x2;
double answer;
double epsilon;
std::map<std::string, std::string> parameters;
};

//Computes the sum of the areas of trapezoids under the curve 
__device__ double Sum(int trapStart, int trapEnd, double* area)
{	
	int i;
	double total = 0;
        if(((trapEnd+1) - trapStart) <= 0)
                return -1;
	else
	{
		for(i = trapStart; i <= trapEnd; i++)
		{
			total += area[i];
		}
	}
	return total;
}

//computes the area of a trapezoid under the curve with the number of trapezoids
//increasing by powers of 2. Then populates an array with the areas of each trapezoid
__global__ void trapAreaPow2(int x1, int x2, double* area, int numTraps)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index < numTraps)
	{
		int n, t;
		double d, x;
		n = index + 2;
		n = log2((double) n);

		if(index == 6 || index == 62)
			n++;

		n = exp2((double) n);
		d = (x2 - x1) / (double)n;
		t = index - n + 3;
		x = ((t - 1) * d) + x1;
		area[index] = ((myCurve(x) + myCurve(x+d)) / 2) * d;
	}
}

//each curve is integrated by trapezoidal sums with the number trapezoids increasing by powers of 2 this method
//calculates the sum of the areas of trapezoids for each set of trapezoids and populates an array with the results
__global__ void trapSumPow2(double* sum, int numSums, double* area)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numSums)
	{
		int n, trapStart, trapEnd;	
		n = exp2((double) (index+1));
		trapStart = n - 2;
		trapEnd = (n * 2) - 3;
		sum[index] = Sum(trapStart, trapEnd, area);
	}
}

//computes the area of a trapezoid under the curve with the number of trapezoids
//increasing linearly. Then populates an array with the areas of each trapezoid
__global__ void trapArea(int x1, int x2, int S, double* area, int numTraps)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numTraps)
	{
		int a, b, t, n, i=0;
		double d, c;
		t = index;
		while(t >= 0)
		{
			t = t - S - i;
			i++;
		}
		a = i - 1;
		n = S + a;
		d = (x2 - x1) / (double) n;
		b = ((a * a) - a) / 2;
		c = index - ((a * S) + b);
		c = x1 + (c * d);
		area[index] =  ((myCurve(c) + myCurve(c+d)) / 2) * d;
	}
}

//each curve is integrated by trapezoidal sums with the number trapezoids increasing linearly this method
//calculates the sum of the areas of trapezoids for each set of trapezoids, and populates an array with the results
__global__ void trapSum(int S, double* sum, int numSums, double* area)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < numSums)
	{
		int trapStart, trapEnd;
		trapStart = (index * S) + (((index*index) - index) / 2);
		trapEnd = trapStart + S + index - 1;
		sum[index] = Sum(trapStart, trapEnd, area);
	}
}
