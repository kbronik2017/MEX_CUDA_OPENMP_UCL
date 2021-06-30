
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense vector-sparse/dense vector dot product  Z=CuMatlab_dot(Sparse/Dense(X),Sparse/Dense(Y)).
 * Z= X.Y
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

#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>

// Input Arguments
#define	INPUTSPARSEA   prhs[0]
#define	INPUTSPARSEB   prhs[1]

// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseSSC(int nlhs, mxArray *plhs[],
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

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be two.";
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be one.";
   if ((nrhs!=2)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }
   if ((nlhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", OutputErrMsg);
    }
 char *input_buf0;
 input_buf0 = mxArrayToString(INPUTSPARSEA);

      if ((mxIsChar(INPUTSPARSEA))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    
 char *input_buf1;
 input_buf1 = mxArrayToString(INPUTSPARSEB);

      if ((mxIsChar(INPUTSPARSEB))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    } 



if (mxIsGPUArray(INPUTSPARSEA) && mxIsGPUArray(INPUTSPARSEB)) {
    
    mxGPUArray const *INPUTSPARSEGPUA;
    mxGPUArray const *INPUTSPARSEGPUB;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTSPARSEGPUA = mxGPUCreateFromMxArray(INPUTSPARSEA);
    INPUTSPARSEGPUB = mxGPUCreateFromMxArray(INPUTSPARSEB);
    
   
	
    if((mxGPUIsSparse(INPUTSPARSEGPUA))&& (mxGPUIsSparse(INPUTSPARSEGPUB)) ){
        
    const mwSize *dimsGPUSA;
    dimsGPUSA=mxGPUGetDimensions(INPUTSPARSEGPUA);
    int numARows, numAColumns;
    numARows = (int)dimsGPUSA[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPUSA[1]; /* gets number of columns of A */
    
    const mwSize *dimsGPUSB;
    dimsGPUSB=mxGPUGetDimensions(INPUTSPARSEGPUB);
    int numBRows, numBColumns;
    numBRows = (int)dimsGPUSB[0]; /* gets number of rows of B */
    numBColumns = (int)dimsGPUSB[1]; /* gets number of columns of B */
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, first/second arguments must be sparse/sparse column/row vector");
             
    }
    
    if ( mxGPUGetNumberOfElements(INPUTSPARSEGPUA)!=mxGPUGetNumberOfElements(INPUTSPARSEGPUB)) {
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);     
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of sparse vector (first argument) must be equal to row/column number of sparse vector(second argument).");
             
    }
      
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); 
          
    mwIndex nnz1;
    
    mxArray * tempx = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUA);
    nnz1 = *(mxGetJc(tempx) + numAColumns);
     //nnz1=(mwSize)ceil(numARows*numAColumns);
    int nnz= static_cast<int> (nnz1);


    int *pointerrow =0;
    mxArray *row_sort;
   if (numAColumns == 1) {
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSetIXY(tempx , pointerrow, nnz);
    }
    
   if (numARows == 1) {

   
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
    
    Jc_GetSetIXY(tempx , pointerrow);
   
        } 
   
    cuDoubleComplex  *pointerval = (cuDoubleComplex *)mxGetComplexDoubles(tempx);
            
   size_t pivot_dimensionsrow[1] = {nnz};
   
   size_t pivot_dimensionsvalue[1] = {nnz};    
      mxGPUArray *row_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortA=(int *)mxGPUGetData(row_sortA);
 gpuErrchk(cudaMemcpy(xrow_sortA, pointerrow, nnz * sizeof(*xrow_sortA), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortA=(cuDoubleComplex*)mxGPUGetData(val_sortA);
 gpuErrchk(cudaMemcpy(xval_sortA, pointerval, nnz * sizeof(*xval_sortA), cudaMemcpyHostToDevice));    
   
         mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
         mxDestroyArray(row_sort);
         mxDestroyArray(tempx);
		 
     mwIndex nnz2;
    mxArray * VLSXY2 = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUB);
    nnz2 = *(mxGetJc(VLSXY2) + numBColumns);
    
    int nnzB= static_cast<int> (nnz2);    
                 
    int *pointerrowB =0;
    mxArray *row_sortB;
   if (numBColumns == 1) {
    row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(VLSXY2 , pointerrowB, nnzB);
    }
    
   if (numBRows == 1) {

   
    row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    pointerrowB = (int *)mxGetInt32s(row_sortB);
    
    Jc_GetSetIXY(VLSXY2 , pointerrowB);
   
        } 
   
    cuDoubleComplex  *pointervalB = (cuDoubleComplex *)mxGetComplexDoubles(VLSXY2);
            
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB=(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortB=(cuDoubleComplex*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice));    
    
    size_t pivot_dimensionsvalueV[1] = {numBRows};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *VALDENSE = (cuDoubleComplex *)mxGPUGetData(VAL);

     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, VALDENSE, 
              CUSPARSE_INDEX_BASE_ONE));

         mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mxDestroyArray(row_sortB);
         mxDestroyArray(VLSXY2);    

 
  cuDoubleComplex VALOUT= make_cuDoubleComplex(0.0, 0.0); 
  cusparseSafeCall(cusparseZdotci(handle, nnz, 
              xval_sortA, 
              xrow_sortA, VALDENSE, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
  
      int nnzx=1;    
      mwSize nnzm=(mwSize)nnzx;
      OUTPUTMATRIX = mxCreateSparse(1,1,nnzm,mxCOMPLEX);
  
      mwIndex *irs = static_cast<mwIndex *> (mxMalloc (nnzx * sizeof(mwIndex)));
      irs[0] = 0;

      mwIndex *jcs = static_cast<mwIndex *> (mxMalloc (2 * sizeof(mwIndex)));
      jcs[0]=jcs[1]=1;
        
      mxComplexDouble* PRS =static_cast<mxComplexDouble *> (mxMalloc (nnzx * sizeof(mxComplexDouble)));
		 PRS[0].real = VALOUT.x;
         PRS[0].imag = VALOUT.y;
		 
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetComplexDoubles (OUTPUTMATRIX)) ;      
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
        mxGPUDestroyGPUArray(row_sortA);
        mxGPUDestroyGPUArray(val_sortA);
        mxGPUDestroyGPUArray(row_sortBB);
        mxGPUDestroyGPUArray(val_sortBB);       
        mxGPUDestroyGPUArray(VAL);
        cusparseDestroyMatDescr(descrA);   
		cusparseDestroy(handle);
  
        
        }
    
        else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }
    
   }
     
////////////////////////////////////////////////////////////////////////////////////  
    else if (!(mxIsGPUArray(INPUTSPARSEA)) && !(mxIsGPUArray(INPUTSPARSEB))){
   
     // if ((mxGetClassID(INPUTSPARSEA) != mxDOUBLE_CLASS) || (mxGetClassID(INPUTSPARSEB) != mxDOUBLE_CLASS)) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  cuDoubleComplex precision.");
             
   // }
    if((mxIsSparse(INPUTSPARSEA))&& (mxIsSparse(INPUTSPARSEB)) ){
    
     mxInitGPU();
    const mwSize *dimsCPUA;
    dimsCPUA=mxGetDimensions(INPUTSPARSEA);
    
    int  numARows = (int)dimsCPUA[0]; /* gets number of rows of A */
    int  numAColumns = (int)dimsCPUA[1]; /* gets number of columns of A */
   
    const mwSize *dimsCPUB;
    dimsCPUB=mxGetDimensions(INPUTSPARSEB);
    
    int  numBRows = (int)dimsCPUB[0]; /* gets number of rows of B */
    int  numBColumns = (int)dimsCPUB[1]; /* gets number of columns of B */
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, first/second arguments must be sparse/sparse column/row vector");
             
    }
    
    if ( mxGetNumberOfElements(INPUTSPARSEA)!=mxGetNumberOfElements(INPUTSPARSEB)) {
    
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of sparse vector (first argument) must be equal to row/column number of sparse vector(second argument).");
             
    }
 
      
    
    mwIndex nnz1;
 
    nnz1 = *(mxGetJc(INPUTSPARSEA) + numAColumns);
    
    int nnz= static_cast<int> (nnz1);
    
    int *pointerrow =0;
    mxArray *row_sort;
   if (numAColumns == 1) {
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSetIXY(INPUTSPARSEA , pointerrow, nnz);
    
    }
    
   if (numARows == 1) {

   
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
    
    Jc_GetSetIXY(INPUTSPARSEA , pointerrow);
   
        }
   
    cuDoubleComplex  *pointerval = (cuDoubleComplex *)mxGetComplexDoubles(INPUTSPARSEA);
            
   size_t pivot_dimensionsrow[1] = {nnz};
   
   size_t pivot_dimensionsvalue[1] = {nnz};    
      mxGPUArray *row_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortA=(int *)mxGPUGetData(row_sortA);
 gpuErrchk(cudaMemcpy(xrow_sortA, pointerrow, nnz * sizeof(*xrow_sortA), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortA=(cuDoubleComplex*)mxGPUGetData(val_sortA);
 gpuErrchk(cudaMemcpy(xval_sortA, pointerval, nnz * sizeof(*xval_sortA), cudaMemcpyHostToDevice));    
   
         
         mxDestroyArray(row_sort);
         
	
	mwIndex nnz2;
 
    nnz2 = *(mxGetJc(INPUTSPARSEB) + numBColumns);
  
    int nnzB= static_cast<int> (nnz2);
    
    int *pointerrowB =0;
    mxArray *row_sortB;
   if (numBColumns == 1) {
    row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(INPUTSPARSEB , pointerrowB, nnzB);
    
    }
    
   if (numBRows == 1) {

   
    row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    pointerrowB = (int *)mxGetInt32s(row_sortB);
    
    Jc_GetSetIXY(INPUTSPARSEB , pointerrowB);
   
        }
   
    cuDoubleComplex  *pointervalB = (cuDoubleComplex *)mxGetComplexDoubles(INPUTSPARSEB);
            
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB=(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortB=(cuDoubleComplex*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice));    
   
         
         mxDestroyArray(row_sortB);    
		 
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
	
   cuDoubleComplex  *VALDENSE;
   mxGPUArray *VAL;
if (numBColumns == 1) {	
    
  size_t   pivot_dimensionsvalueV[1] = {numBRows};
  VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (cuDoubleComplex *)mxGPUGetData(VAL);
    
     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, VALDENSE, 
              CUSPARSE_INDEX_BASE_ONE));
    
    }  

if (numBRows == 1) {
	
   size_t  pivot_dimensionsvalueV[1] = {numBColumns};
   VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (cuDoubleComplex *)mxGPUGetData(VAL);
    
     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, VALDENSE, 
              CUSPARSE_INDEX_BASE_ONE));
  
} 	  
	  

   cuDoubleComplex VALOUT= make_cuDoubleComplex(0.0, 0.0); 
  cusparseSafeCall(cusparseZdotci(handle, nnz, 
              xval_sortA, 
              xrow_sortA, VALDENSE, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
  
      int nnzx=1;    
      mwSize nnzm=(mwSize)nnzx;
      OUTPUTMATRIX = mxCreateSparse(1,1,nnzm,mxCOMPLEX);
  
      mwIndex *irs = static_cast<mwIndex *> (mxMalloc (nnzx * sizeof(mwIndex)));
      irs[0] = 0;

      mwIndex *jcs = static_cast<mwIndex *> (mxMalloc (2 * sizeof(mwIndex)));
      jcs[0]=jcs[1]=1;
        
      mxComplexDouble* PRS =static_cast<mxComplexDouble *> (mxMalloc (nnzx * sizeof(mxComplexDouble)));
		 PRS[0].real = VALOUT.x;
         PRS[0].imag = VALOUT.y;
		 
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetComplexDoubles (OUTPUTMATRIX)) ;      
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
        mxGPUDestroyGPUArray(row_sortA);
        mxGPUDestroyGPUArray(val_sortA);
        mxGPUDestroyGPUArray(row_sortBB);
        mxGPUDestroyGPUArray(val_sortBB);      
        mxGPUDestroyGPUArray(VAL);
        cusparseDestroyMatDescr(descrA);  
		cusparseDestroy(handle);
    
    }
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }
    
 }
        //
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
