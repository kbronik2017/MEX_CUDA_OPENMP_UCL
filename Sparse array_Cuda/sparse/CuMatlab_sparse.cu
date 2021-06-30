
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0],prhs[1],prhs[2]  := mxGPUArray or CPU Array}[double or complex double]
 * Create sparse matrix  
 * Z=CuMatlab_sparse(X) 
 * Z=CuMatlab_sparse(X,Y)
 * Z=CuMatlab_sparse(X,Y,Z)
 * Z=CuMatlab_sparse(X,Y,Z,row,column) 
 * Z=CuMatlab_sparse(X,Y,Z,row,column,nz)
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

#include "CuMatlab_sparseO.cu"
#include "CuMatlab_sparseZ.cu"
#include "CuMatlab_sparseD.cu"
#include "CuMatlab_sparseDX.cu"
#include "CuMatlab_sparseDY.cu"
#include "CuMatlab_sparseZX.cu"
#include "CuMatlab_sparseZY.cu"
#include <cuda.h>
#include <cuda_runtime.h>


extern "C" static void mexCuMatlab_sparseZ(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseD(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseO(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);                 
 
extern "C" static void mexCuMatlab_sparseDX(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
       
extern "C" static void mexCuMatlab_sparseDY(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);              
                 
extern "C" static void mexCuMatlab_sparseZX(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseZY(int nlhs, mxArray *plhs[],
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
   if (nrhs==1) {
        mexCuMatlab_sparseO(nlhs, plhs,
                 nrhs, prhs);
                 return;
    }
    
   else  if (nrhs==2) {
        plhs[0] = mxCreateSparse((mwSize)mxGetScalar(prhs[0]),(mwSize)mxGetScalar(prhs[1]),0,mxREAL);
                 return;
    }  
    
   else if (nrhs==3) {

        if (mxIsGPUArray(prhs[2])) {
        
    mxGPUArray const *tempGPU;
    tempGPU = mxGPUCreateFromMxArray(prhs[2]);
        
       if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
           
       mexCuMatlab_sparseDX(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU);
       return;
           }
       else if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS  && mxGPUGetComplexity(tempGPU) == mxCOMPLEX){ 
              mexCuMatlab_sparseZX(nlhs, plhs,
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
     else if(!mxIsGPUArray(prhs[2])) {  
       if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (!mxIsComplex(prhs[2]))){ 
            
           mexCuMatlab_sparseDX(nlhs, plhs,
                 nrhs, prhs);
                 return; 
       
           }
       else if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (mxIsComplex(prhs[2]))){ 
           
           mexCuMatlab_sparseZX(nlhs, plhs,
                 nrhs, prhs);
                 return; 
           }
           
           
       else{  
               
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
           
       }           
                 
    }
    
    else if (nrhs==5) {
    if (mxIsGPUArray(prhs[2])) {
        
    mxGPUArray const *tempGPU;
    tempGPU = mxGPUCreateFromMxArray(prhs[2]);
        
       if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
           
       mexCuMatlab_sparseD(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU);
       return;
           }
       else if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS  && mxGPUGetComplexity(tempGPU) == mxCOMPLEX){ 
              mexCuMatlab_sparseZ(nlhs, plhs,
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
     else if(!mxIsGPUArray(prhs[2])) {  
       if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (!mxIsComplex(prhs[2]))){ 
            
           mexCuMatlab_sparseD(nlhs, plhs,
                 nrhs, prhs);
                 return; 
       
           }
       else if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (mxIsComplex(prhs[2]))){ 
           
           mexCuMatlab_sparseZ(nlhs, plhs,
                 nrhs, prhs);
                 return; 
           }
           
           
       else{  
               
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
           
       }
    }
   else if (nrhs==6) {

    if (mxIsGPUArray(prhs[2])) {
        
    mxGPUArray const *tempGPU;
    tempGPU = mxGPUCreateFromMxArray(prhs[2]);
        
       if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS && mxGPUGetComplexity(tempGPU) == mxREAL){ 
           
       mexCuMatlab_sparseDY(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU);
       return;
           }
       else if (mxGPUGetClassID(tempGPU) == mxDOUBLE_CLASS  && mxGPUGetComplexity(tempGPU) == mxCOMPLEX){ 
              mexCuMatlab_sparseZY(nlhs, plhs,
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
     else if(!mxIsGPUArray(prhs[2])) {  
      if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (!mxIsComplex(prhs[2]))){ 
            
           mexCuMatlab_sparseDY(nlhs, plhs,
                 nrhs, prhs);
                 return; 
       
           }
       else if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS && (mxIsComplex(prhs[2]))){ 
           
           mexCuMatlab_sparseZY(nlhs, plhs,
                 nrhs, prhs);
                 return; 
           }
           
           
       else{  
               
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
           
           }
           
       }

      }  
    
           else if  ((nrhs!=1) || (nrhs!=2)  ||  (nrhs!=3)  || (nrhs!=5) || (nrhs!=6)) {
            
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input/output arguments! input arguments must be one/two/three/five or six \n"); 
                return;    
        }

}
