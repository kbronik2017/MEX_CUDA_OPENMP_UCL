
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double or complex double]
 * This  computes the singular value decomposition (SVD) of sparse/dense matrix  [U, S, V]=CuMatlab_svd(Sparse/Dense(X))
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cusparse_v2.h>

#include "CuMatlab_svdD.cu"
#include "CuMatlab_svdZ.cu"

#include <cuda.h>
#include <cuda_runtime.h>


extern "C" static void mexCuMatlab_svdD(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_svdZ(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 

                 
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
 int nDevices;
cudaError_t errCode =cudaGetDeviceCount(&nDevices); 
//int nDevices;
//cudaGetDeviceCount(&nDevices);

if (errCode != cudaSuccess){
printf("Error! No CUDA devices found! \n");
return;
}

    
   if (nrhs==1 && nlhs==3) {

        if (mxIsGPUArray(prhs[0])) {
        
    mxGPUArray const *tempGPU;
    tempGPU = mxGPUCreateFromMxArray(prhs[0]);
        
       if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
           
       mexCuMatlab_svdD(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU);
       return;
           }
       else if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS  && mxGPUGetComplexity(tempGPU) == mxCOMPLEX){ 
              mexCuMatlab_svdZ(nlhs, plhs,
                 nrhs, prhs);
       
       mxGPUDestroyGPUArray(tempGPU);
       return;
           
           }
           
       else{ 
               
       mxGPUDestroyGPUArray(tempGPU);
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
    }
    //
     else if(!mxIsGPUArray(prhs[0])) {  
       if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS && (!mxIsComplex(prhs[0]))){ 
            
           mexCuMatlab_svdD(nlhs, plhs,
                 nrhs, prhs);
                 return; 
       
           }
       else if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS && (mxIsComplex(prhs[0]))){ 
           
           mexCuMatlab_svdZ(nlhs, plhs,
                 nrhs, prhs);
                 return; 
           }
           
           
       else{  
               
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
           
       }           
                 
    }
    
 
    
     else if  ((nrhs<1) || (nrhs>1)  ||  (nlhs<3)  || (nlhs>3) ) {
            
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input/output arguments! input argument must be one and output arguments must be three\n"); 
                return;    
        }

}
