# GPUTrapezoid
# Language: CUDA
# Input: TXT
# Output: SCREEN
# Tested with: PluMA 1.0, CUDA 10

Approximate the solution of a function with a trapezoidal sum on the GPU
Determine the minimum number needed to arrive within an epsilon of the real solution

Original authors: Jose Nunez, Bhavyta Chauhan, Nicolas Dabdoub, Rafael Leal

The plugin accepts as input a TXT file of tab-delimited keyword-value pairs:
nTrapsPow2: number of Trapezoids
nSumsPow2: number of Sums
x1: starting x value
x2: ending x value
answer: actual answer
epsilon: Epsilon value

The function to integrate: function.h.  Future goal is to customize this.

Number of trapezoids will be printed to the screen.
