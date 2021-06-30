
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense vector-sparse/dense vector addition   Z=CuMatlab_addV(Sparse/Dense(X),Sparse/Dense(Y), alpha).
 * Z= alpha*X+Y
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

#include "CuMatlab_sparseSSR.cu"
#include "CuMatlab_sparseSSC.cu"
#include "CuMatlab_sparseDDR.cu"
#include "CuMatlab_sparseDDC.cu"
#include "CuMatlab_sparseSDR.cu"
#include "CuMatlab_sparseSDC.cu"
#include "CuMatlab_sparseDSR.cu"
#include "CuMatlab_sparseDSC.cu"
#include <cuda.h>
#include <cuda_runtime.h>


extern "C" static void mexCuMatlab_sparseSSR(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseSSC(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseDDR(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);
                 
extern "C" static void mexCuMatlab_sparseDDC(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);  
                 
extern "C" static void mexCuMatlab_sparseSDR(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]);                 
                 
extern "C" static void mexCuMatlab_sparseSDC(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]); 
                 
extern "C" static void mexCuMatlab_sparseDSR(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]); 
                 
extern "C" static void mexCuMatlab_sparseDSC(int nlhs, mxArray *plhs[],
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

    
   if (nrhs==3 && nlhs==1) {

        if (mxIsGPUArray(prhs[0]) &&  mxIsGPUArray(prhs[1])) {
        
    mxGPUArray const *tempGPU1;
    tempGPU1 = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *tempGPU2;
    tempGPU2 = mxGPUCreateFromMxArray(prhs[1]); 
    mxGPUArray const *tempGPU3;
    tempGPU3 = mxGPUCreateFromMxArray(prhs[2]); 

       if ((mxGPUGetClassID(tempGPU1) == mxDOUBLE_CLASS) && (mxGPUGetComplexity(tempGPU1) == mxREAL) && (mxGPUGetClassID(tempGPU2) == mxDOUBLE_CLASS) && (mxGPUGetComplexity(tempGPU2) == mxREAL) ){ 
         if ( mxGPUGetComplexity(tempGPU3) != mxREAL ) {
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);

               mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input(THIRD AND FOURTH ARGUMENTS) must be scalar and double precision. %s\n");
                return;
               }  
           
       if ( (mxGPUIsSparse(tempGPU1))&& (mxGPUIsSparse(tempGPU2))) {
     
           
       mexCuMatlab_sparseSSR(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       

       return;
       }
       if ( (!mxGPUIsSparse(tempGPU1))&& (!mxGPUIsSparse(tempGPU2))) {
     
           
       mexCuMatlab_sparseDDR(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       

       return;
       }
      if ( (mxGPUIsSparse(tempGPU1))&& (!mxGPUIsSparse(tempGPU2))) {
     
           
       mexCuMatlab_sparseSDR(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
       }
      if ( (!mxGPUIsSparse(tempGPU1))&& (mxGPUIsSparse(tempGPU2))) {
     
           
       mexCuMatlab_sparseDSR(nlhs, plhs,
                 nrhs, prhs);
           
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
       }
       
           }
       else if ((mxGPUGetClassID(tempGPU1) == mxDOUBLE_CLASS)  && (mxGPUGetComplexity(tempGPU1) == mxCOMPLEX) && (mxGPUGetClassID(tempGPU2) == mxDOUBLE_CLASS)  && (mxGPUGetComplexity(tempGPU2) == mxCOMPLEX) ){ 

           if ( mxGPUGetComplexity(tempGPU3) != mxCOMPLEX ){
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
               mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input(THIRD AND FOURTH ARGUMENTS) must be complex and double precision. %s\n");
                return;
               }   
               
         if ( (mxGPUIsSparse(tempGPU1))&& (mxGPUIsSparse(tempGPU2))){
     
               
        mexCuMatlab_sparseSSC(nlhs, plhs,
                 nrhs, prhs);    
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
           }
        if ( (!mxGPUIsSparse(tempGPU1))&& (!mxGPUIsSparse(tempGPU2))){
     
               
        mexCuMatlab_sparseDDC(nlhs, plhs,
                 nrhs, prhs);    
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
           }          
        if ( (mxGPUIsSparse(tempGPU1))&& (!mxGPUIsSparse(tempGPU2))){
     
               
        mexCuMatlab_sparseSDC(nlhs, plhs,
                 nrhs, prhs);    
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
           }        
        if ( (!mxGPUIsSparse(tempGPU1))&& (mxGPUIsSparse(tempGPU2))){
     
               
        mexCuMatlab_sparseDSC(nlhs, plhs,
                 nrhs, prhs);    
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       return;
           } 
           
           
         }
           
       else{ 
               
       mxGPUDestroyGPUArray(tempGPU1);
       mxGPUDestroyGPUArray(tempGPU2);
       mxGPUDestroyGPUArray(tempGPU3);
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
            
           }
    }
    //
     else if(!mxIsGPUArray(prhs[0]) && !mxIsGPUArray(prhs[1])) { 
     mxGPUArray const *tempGPU3;
    tempGPU3 = mxGPUCreateFromMxArray(prhs[2]);

       if ((mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) && (!mxIsComplex(prhs[0])) && (mxGetClassID(prhs[1]) == mxDOUBLE_CLASS) && (!mxIsComplex(prhs[1]))){ 
            
       if ( mxGPUGetComplexity(tempGPU3) != mxREAL ) {
       
       mxGPUDestroyGPUArray(tempGPU3);
       
               mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input(THIRD AND FOURTH ARGUMENTS) must be scalar and double precision. %s\n");
                return;
               } 
           
           
           
           if ( (mxIsSparse(prhs[0]))&& (mxIsSparse(prhs[1]))) {
       
           
            mexCuMatlab_sparseSSR(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
       }
           if ( (!mxIsSparse(prhs[0]))&& (!mxIsSparse(prhs[1]))) {
       
           
            mexCuMatlab_sparseDDR(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
       }
           if ( (mxIsSparse(prhs[0]))&& (!mxIsSparse(prhs[1]))) {
       
           
            mexCuMatlab_sparseSDR(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
                        
                 return; 
       }
           if ( (!mxIsSparse(prhs[0]))&& (mxIsSparse(prhs[1]))) {
       
           
            mexCuMatlab_sparseDSR(nlhs, plhs,
                 nrhs, prhs);
        mxGPUDestroyGPUArray(tempGPU3);
                        
                 return; 
       }
       
       
           }
       else if ((mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) && (mxIsComplex(prhs[0])) && (mxGetClassID(prhs[1]) == mxDOUBLE_CLASS) && (mxIsComplex(prhs[1]))){ 
           if ( mxGPUGetComplexity(tempGPU3) != mxCOMPLEX ){
       
       mxGPUDestroyGPUArray(tempGPU3);
       
               mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments!, input(THIRD AND FOURTH ARGUMENTS) must be complex and double precision. %s\n");
                return;
               }
               
               
               
        if ( (mxIsSparse(prhs[0]))&& (mxIsSparse(prhs[1]))){
       
               
           mexCuMatlab_sparseSSC(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
           }
        if ( (!mxIsSparse(prhs[0]))&& (!mxIsSparse(prhs[1]))){
       
               
           mexCuMatlab_sparseDDC(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
           }
        if ( (mxIsSparse(prhs[0]))&& (!mxIsSparse(prhs[1]))){
       
               
           mexCuMatlab_sparseSDC(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
           }
         if ( (!mxIsSparse(prhs[0]))&& (mxIsSparse(prhs[1]))){
       
               
           mexCuMatlab_sparseDSC(nlhs, plhs,
                 nrhs, prhs);
       mxGPUDestroyGPUArray(tempGPU3);
       
                 return; 
           }
           
         }  
           
       else{  

       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
            
           }
           
       }           
                 
    }
    
 
    
     else if  ((nrhs<3) || (nrhs>3)  ||  (nlhs<1)  || (nlhs>1) ) {
            
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input/output arguments! input arguments must be three and output argument must be one\n"); 
                return;
        }

}
