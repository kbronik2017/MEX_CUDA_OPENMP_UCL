
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>


// Input Arguments
#define	ROW      prhs[0]
#define	COLUMN   prhs[1]
#define	VALUE    prhs[2]
#define	NROWS    prhs[3]
#define	NCOLS    prhs[4]



// Output Arguments
#define	OUTPUTMATRIX   plhs[0]

 
   
    
extern "C" static void mexCuMatlab_sparseD(int nlhs, mxArray *plhs[],
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

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be five.";
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be one.";
   if ((nrhs!=5)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }
   if ((nlhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", OutputErrMsg);
    }

 char *input_buf0;
 input_buf0 = mxArrayToString(ROW);

      if ((mxIsChar(ROW))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    
     char *input_buf1;
 input_buf1 = mxArrayToString(COLUMN);

      if ((mxIsChar(COLUMN))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    }
    
     char *input_buf2;
 input_buf2 = mxArrayToString(VALUE);

      if ((mxIsChar(VALUE))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(THIRD ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf2);
    }

if (mxIsGPUArray(ROW)  && mxIsGPUArray(COLUMN) && mxIsGPUArray(VALUE) ) {

     mxInitGPU();
    
   mxGPUArray const *ROWGPU;
   ROWGPU= mxGPUCreateFromMxArray(ROW);
   
    
   mxGPUArray const *COLUMNGPU;
   COLUMNGPU= mxGPUCreateFromMxArray(COLUMN);
    
   
   mxGPUArray const *VALUEGPU;
   VALUEGPU= mxGPUCreateFromMxArray(VALUE);
   
   if((mxGPUIsSparse(ROWGPU)==1)|| (mxGPUIsSparse(COLUMNGPU)==1) || (mxGPUIsSparse(VALUEGPU)==1)){
       //plhs[0] = mxGPUCreateMxArrayOnGPU(INPUTMATRIXGPUx);
       printf("Warning! Input(FIRST, SECOND and THIRD ARGUMENTS) must be non sparse! \n");  
      // mxGPUDestroyGPUArray(INPUTMATRIXGPUx);
      mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
                return;
    
    }
   
   if ( (mxGPUGetClassID(ROWGPU) != mxDOUBLE_CLASS)  && (mxGPUGetClassID(COLUMNGPU) != mxDOUBLE_CLASS) && (mxGPUGetClassID(VALUEGPU) != mxDOUBLE_CLASS)) {
      mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
       
       mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST, SECOND AND THIRD ARGUMENTS) must be  double precision.");
    }
    if ( (mxGPUGetComplexity(ROWGPU) != mxREAL)  || (mxGPUGetComplexity(COLUMNGPU) != mxREAL) || (mxGPUGetComplexity(VALUEGPU) != mxREAL)) {
      mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU); 
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST, SECOND AND THIRD ARGUMENTS) must be real with no imaginary components.");
    }
    if ( !(mxIsScalar(NROWS)) || !(mxIsScalar(NCOLS))) {
      mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input (FOURTH and  FIFTH ARGUMENTS) must be scalar.");
             
    }
    int NrowsA= (int)mxGetScalar(NROWS);   
      
    int NcolsA= (int)mxGetScalar(NCOLS);
    
    int nnzR= static_cast<int> (mxGPUGetNumberOfElements(ROWGPU));
    int nnzC= static_cast<int> (mxGPUGetNumberOfElements(COLUMNGPU));
    int nnzV= static_cast<int> (mxGPUGetNumberOfElements(VALUEGPU));
    int NNZMAXA=nnzV;
    if ( (nnzR!= nnzC) || (nnzC != nnzV)|| (nnzR != nnzV) ) {
	  mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input vectors (FIRST, SECOND, THIRD ARGUMENTS) must be the same lengths.");
             
    }
   if ( nnzR>(NrowsA*NcolsA) ) {
	  mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, index exceeds array bounds [number of non zero greater than matrix dimensions (row*column)].");
             
    }
    
    mxArray * RW=mxGPUCreateMxArrayOnCPU(ROWGPU);
	mxArray * CL =mxGPUCreateMxArrayOnCPU(COLUMNGPU);
    mxArray * VL=mxGPUCreateMxArrayOnCPU(VALUEGPU);
    
   // int *nrr=(int *)malloc(nnzR * sizeof(*nrr));
   // int *nrc=(int *)malloc(nnzR * sizeof(*nrc));
	    std::vector<MATRIX> vect;
        int j;
      #pragma omp for  schedule(static) nowait
         for ( j = 0; j < nnzR; ++j) {
           vect.push_back(MATRIX((static_cast<int> ((mxGetDoubles(RW))[j])),  (static_cast<int> ((mxGetDoubles(CL))[j])), (static_cast<double> ((mxGetDoubles(VL))[j])))); 
         //  nrr[i]=(static_cast<int> ((mxGetPr(RW))[i]));
         //  nrc[i]=(static_cast<int> ((mxGetPr(CL))[i]));
           }
            
  // int  Nr= *std::max_element(nrr, nrr + nnzR, max_elem);
  // int  Nc= *std::max_element(nrc, nrc + nnzR, max_elem);  
           
	std::sort(vect.begin(), vect.end());
	std::vector<MATRIX> vect_temp; 
    vect_temp= vect;
    	
    int i = 0;
	     
	std::vector<MATRIX>::iterator ity = vect.begin();
//	#pragma omp parallel for  shared(nnzR) private(i)
	for (std::vector<MATRIX>::iterator itx = vect.begin(); itx != vect.end(); itx++){
		
		
		ity = itx + 1;
		while (ity != vect.end())
		{


			//for (ity ; ity != vectx.end(); ity++){

			if (itx->row_C == ity->row_C && itx->column_C == ity->column_C){
				vect_temp[i].value_C = vect_temp[i].value_C + ity->value_C;

				vect_temp[std::distance(vect.begin(), ity)].checked = true;

			}
			ity++;
				//}
		}
		i++;
		
	}
    //    #pragma omp parallel for 
  for (auto it = vect_temp.begin(); it != vect_temp.end();) {
		if ((it->checked==true) ||(it->value_C==0)) {
			it = vect_temp.erase(it);
		}
		else {
			++it;
		}
	}
  nnzR=nnzC=nnzV= (int)vect_temp.size();
  
   mxArray * ROWx =mxCreateNumericMatrix(nnzR, 1, mxINT32_CLASS, mxREAL);
    int *h_A_RowIndices_coo = (int *)mxGetInt32s(ROWx);
    
           #pragma omp parallel for shared(nnzR) private(i)
         for (i = 0; i < nnzR; ++i) {
          // h_A_RowIndices_coo[i] = static_cast<int> ((mxGetPr(RW))[i]); 
           h_A_RowIndices_coo[i] =vect_temp[i].row_C;
            }
            
   mxArray * COLUMNx =mxCreateNumericMatrix(nnzC, 1, mxINT32_CLASS, mxREAL);
    int *h_A_ColIndices_coo = (int *)mxGetInt32s(COLUMNx);
    
           #pragma omp parallel for shared(nnzC) private(i)
         for (i = 0; i < nnzC; ++i) {
          // h_A_ColIndices_coo[i] = static_cast<int> ((mxGetPr(CL))[i]);
           h_A_ColIndices_coo[i] = vect_temp[i].column_C;
            }
    
    
   mxArray * VALUEx =mxCreateNumericMatrix(nnzV, 1, mxDOUBLE_CLASS, mxREAL);
    double *h_A1_coo = (double *)mxGetDoubles(VALUEx); 
    
         #pragma omp parallel for shared(nnzV) private(i)
         for (i = 0; i < nnzV; ++i) {
          // h_A1_coo[i] = static_cast<double> ((mxGetPr(VL))[i]); 
           h_A1_coo[i] = vect_temp[i].value_C;
            }

    int  Nr= *std::max_element(h_A_RowIndices_coo, h_A_RowIndices_coo + nnzR, max_elem);
   
    int  Nc= *std::max_element(h_A_ColIndices_coo, h_A_ColIndices_coo + nnzC, max_elem);          

   if ( (Nr>NrowsA) || (Nc>NcolsA) ) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, index exceeds array bounds: max(first vector)> fourth argument's value or max(second vector)> fifth argument's value .");
             
    }
	//NrowsA= h_A_RowIndices_coo[nnzR-1];    
       
    //NcolsA= Nc;
//////
  	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

     
	//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzV * sizeof(*d_A)));
	//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (NrowsA + 1) * sizeof(*d_A_RowIndices)));
	//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzV * sizeof(*d_A_ColIndices)));
	//int *d_cooRowIndA;       gpuErrchk(cudaMalloc(&d_cooRowIndA, nnzV * sizeof(*d_cooRowIndA)));
	
   size_t pivot_dimensA[1] = {nnzV};
   size_t pivot_dimensROW_A[1] = {NrowsA + 1};
   size_t pivot_dimensCOL_A[1] = {nnzV};
   size_t pivot_dimensCOO_A[1] = {nnzV};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);
    mxGPUArray * COO_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndA = (int *)mxGPUGetData(COO_A); 
	
	
	// --- Descriptor for sparse matrix A
	gpuErrchk(cudaMemcpy(d_A, h_A1_coo, nnzV * sizeof(*d_A), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_ColIndices, h_A_ColIndices_coo, nnzV * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndA, h_A_RowIndices_coo, nnzV * sizeof(*d_cooRowIndA), cudaMemcpyHostToDevice));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);

	int *Pa = NULL;
	void *pBuffera = NULL;
	size_t pBufferSizeInBytesa = 0;
	cusparseXcoosort_bufferSizeExt(handle, NrowsA, NcolsA,
		nnzV,
		d_cooRowIndA,
		d_A_ColIndices, &pBufferSizeInBytesa);

	gpuErrchk(cudaMalloc(&pBuffera, sizeof(char)*pBufferSizeInBytesa));
	gpuErrchk(cudaMalloc(&Pa, sizeof(int)*nnzV));
	cusparseCreateIdentityPermutation(handle, nnzV, Pa);
	cusparseSafeCall(cusparseXcoosortByRow(handle, NrowsA, NcolsA,
		nnzV,
		d_cooRowIndA,
		d_A_ColIndices,
		Pa,
		pBuffera));

	cusparseSafeCall(cusparseDgthr(handle, nnzV, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzV,
		NrowsA,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));


   size_t pivot_dimensionsrow[1] = {nnzR};
   size_t pivot_dimensionscolumn[1] = {NcolsA+1}; 
   size_t pivot_dimensionsvalue[1] = {nnzV};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORT = (double *)mxGPUGetData(VAL_SORT1);

	cusparseSafeCall(cusparseDcsr2csc(handle, NrowsA, NcolsA, nnzV, d_A, d_A_RowIndices, d_A_ColIndices, VALSORT, ROWSORT, COLSORT, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE));

	//gpuErrchk(cudaFree(d_A));
	//gpuErrchk(cudaFree(d_A_RowIndices));
	//gpuErrchk(cudaFree(d_A_ColIndices));
	//gpuErrchk(cudaFree(d_cooRowIndA));
	
	mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(ROW_A);
    mxGPUDestroyGPUArray(COL_A);
    mxGPUDestroyGPUArray(COO_A);
	
	
	
	gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
    
    
   mwSize nnzm=(mwSize)nnzV;
   OUTPUTMATRIX = mxCreateSparse(NrowsA,NcolsA,nnzm,mxREAL);
   
    
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  

        irs = static_cast<mwIndex *> (mxMalloc (nnzR * sizeof(mwIndex)));
       #pragma omp parallel for shared(nnzR) private(i)
         for ( i = 0; i < nnzR; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i])-1; 
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((NcolsA+1) * sizeof(mwIndex)));
        int nc1= NcolsA+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i])-1;
            }
             
           
        mxDouble* PRS = (mxDouble*) mxMalloc (nnzV * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnzV * sizeof(mxDouble), cudaMemcpyDeviceToHost));

           
    
        
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
        mxSetNzmax(OUTPUTMATRIX, (static_cast<mwSize>(NNZMAXA)));
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);

         mxDestroyArray(RW);
         mxDestroyArray(CL);
         mxDestroyArray(VL);
      mxDestroyArray(ROWx);
      mxDestroyArray(COLUMNx);
      mxDestroyArray(VALUEx);
      
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      mxGPUDestroyGPUArray(ROWGPU);
      mxGPUDestroyGPUArray(COLUMNGPU);
      mxGPUDestroyGPUArray(VALUEGPU);
      cusparseDestroyMatDescr(descrA);	  
      cusparseDestroy(handle);
   
   }
     
////////////////////////////////////////////////////////////////////////////////////  
    else if (!(mxIsGPUArray(ROW))  && !(mxIsGPUArray(COLUMN)) && !(mxIsGPUArray(VALUE))){

   if((mxIsSparse(ROW)) || (mxIsSparse(COLUMN))  || (mxIsSparse(VALUE))) {
    
   
       printf("Warning! Input(FIRST ARGUMENT) must be non sparse!\n");   
                return;
        
    } 
            
    if ( !(mxIsScalar(NROWS)) || !(mxIsScalar(NCOLS))) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input (FOURTH and  FIFTH ARGUMENTS) must be scalar.");
             
    }       
            
    int NrowsA= (int)mxGetScalar(NROWS);   
      
    int NcolsA= (int)mxGetScalar(NCOLS);
    
    int nnzR= (int)mxGetNumberOfElements(ROW);
    int nnzC= (int)mxGetNumberOfElements(COLUMN);
    int nnzV= (int)mxGetNumberOfElements(VALUE);
    int NNZMAXA=nnzV;
   if ( (nnzR!= nnzC) || (nnzC != nnzV)|| (nnzR != nnzV) ) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input vectors (FIRST, SECOND, THIRD ARGUMENTS) must be the same lengths.");
             
    }  
    if ( nnzR>(NrowsA*NcolsA) ) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, index exceeds array bounds [number of non zero greater than matrix dimensions (row*column)].");
             
    }
   if ( (mxGetClassID(ROW) != mxDOUBLE_CLASS) || (mxGetClassID(COLUMN) != mxDOUBLE_CLASS)||(mxGetClassID(VALUE) != mxDOUBLE_CLASS)) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST, SECOND AND THIRD ARGUMENTS) must be  double precision.");
             
    }
    if ( (mxIsComplex(ROW))  || (mxIsComplex(COLUMN)) || (mxIsComplex(VALUE))) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST, SECOND AND THIRD ARGUMENTS) must be real with no imaginary components.");
    } 
    
    mxInitGPU();

    
    //int *nrr=(int *)malloc(nnzR * sizeof(*nrr));
   // int *nrc=(int *)malloc(nnzR * sizeof(*nrc));
    
    std::vector<MATRIX> vect;
         int j;
	#pragma omp for schedule(static) nowait
         for ( j = 0; j < nnzR; ++j) {
           vect.push_back(MATRIX((static_cast<int> ((mxGetDoubles(ROW))[j])),  (static_cast<int> ((mxGetDoubles(COLUMN))[j])), (static_cast<double> ((mxGetDoubles(VALUE))[j])))); 
          // nrr[i]=(static_cast<int> ((mxGetPr(ROW))[i]));
         //  nrc[i]=(static_cast<int> ((mxGetPr(COLUMN))[i]));
            }
            
  // int  Nr= *std::max_element(nrr, nrr + nnzR, max_elem);
   
  // int  Nc= *std::max_element(nrc, nrc + nnzR, max_elem);  

   
            
	std::sort(vect.begin(), vect.end());
	std::vector<MATRIX> vect_temp; 
    vect_temp= vect;
    	
    int i = 0;
	     
	std::vector<MATRIX>::iterator ity = vect.begin();
	//#pragma omp parallel for default(none) shared(vect,vect_temp )
	for (std::vector<MATRIX>::iterator itx = vect.begin(); itx != vect.end(); itx++){
		
		
		ity = itx + 1;
		while (ity != vect.end())
		{


			//for (ity ; ity != vectx.end(); ity++){

			if (itx->row_C == ity->row_C && itx->column_C == ity->column_C){
				vect_temp[i].value_C = vect_temp[i].value_C + ity->value_C;

				vect_temp[std::distance(vect.begin(), ity)].checked = true;

			}
			ity++;
				//}
		}
		i++;
		
	}
    
   // #pragma omp parallel for default(none) shared(vect_temp)
   
  for (auto it = vect_temp.begin(); it != vect_temp.end();) {
		if ((it->checked==true) ||(it->value_C==0)) {
			it = vect_temp.erase(it);
		}
		else {
			++it;
		}
	}
  nnzR=nnzC=nnzV= (int) vect_temp.size();

    
   mxArray * ROWx =mxCreateNumericMatrix(nnzR, 1, mxINT32_CLASS, mxREAL);
    int *h_A_RowIndices_coo = (int *)mxGetInt32s(ROWx);
    
           #pragma omp parallel for shared(nnzR) private(i)
         for (i = 0; i < nnzR; ++i) {
           //h_A_RowIndices_coo[i] = static_cast<int> ((mxGetPr(ROW))[i]);
           h_A_RowIndices_coo[i] =vect_temp[i].row_C;
            }
   mxArray * COLUMNx =mxCreateNumericMatrix(nnzC, 1, mxINT32_CLASS, mxREAL);
    int *h_A_ColIndices_coo = (int *)mxGetInt32s(COLUMNx);
           #pragma omp parallel for shared(nnzC) private(i)
         for (i = 0; i < nnzC; ++i) {
           //h_A_ColIndices_coo[i] = static_cast<int> ((mxGetPr(COLUMN))[i]); 
         h_A_ColIndices_coo[i] = vect_temp[i].column_C; 
            }
    
    
   mxArray * VALUEx =mxCreateNumericMatrix(nnzV, 1, mxDOUBLE_CLASS, mxREAL);
    double *h_A1_coo = (double *)mxGetDoubles(VALUEx);    
         #pragma omp parallel for shared(nnzV) private(i)
         for ( i = 0; i < nnzV; ++i) {
           //h_A1_coo[i] =static_cast<double> ((mxGetPr(VALUE))[i]);
           h_A1_coo[i] = vect_temp[i].value_C;
            }
    
            
    int  Nr= *std::max_element(h_A_RowIndices_coo, h_A_RowIndices_coo + nnzR, max_elem);
   
    int  Nc= *std::max_element(h_A_ColIndices_coo, h_A_ColIndices_coo + nnzC, max_elem);          

   if ( (Nr>NrowsA) || (Nc>NcolsA) ) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, index exceeds array bounds: max(first vector)> fourth argument's value or max(second vector)> fifth argument's value .");
             
    }
    // NrowsA= h_A_RowIndices_coo[nnzR-1];    
       
   //  NcolsA= Nc; 
////////////
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	

	//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzV * sizeof(*d_A)));
	//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (NrowsA + 1) * sizeof(*d_A_RowIndices)));
	//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzV * sizeof(*d_A_ColIndices)));
	//int *d_cooRowIndA;       gpuErrchk(cudaMalloc(&d_cooRowIndA, nnzV * sizeof(*d_cooRowIndA)));
	
   size_t pivot_dimensA[1] = {nnzV};
   size_t pivot_dimensROW_A[1] = {NrowsA + 1};
   size_t pivot_dimensCOL_A[1] = {nnzV};
   size_t pivot_dimensCOO_A[1] = {nnzV};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);
    mxGPUArray * COO_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndA = (int *)mxGPUGetData(COO_A); 
	
	
	
	gpuErrchk(cudaMemcpy(d_A, h_A1_coo, nnzV * sizeof(*d_A), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_ColIndices, h_A_ColIndices_coo, nnzV * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndA, h_A_RowIndices_coo, nnzV * sizeof(*d_cooRowIndA), cudaMemcpyHostToDevice));


	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);

	int *Pa = NULL;
	void *pBuffera = NULL;
	size_t pBufferSizeInBytesa = 0;
	cusparseXcoosort_bufferSizeExt(handle, NrowsA, NcolsA,
		nnzV,
		d_cooRowIndA,
		d_A_ColIndices, &pBufferSizeInBytesa);

	gpuErrchk(cudaMalloc(&pBuffera, sizeof(char)*pBufferSizeInBytesa));
	gpuErrchk(cudaMalloc(&Pa, sizeof(int)*nnzV));
	cusparseCreateIdentityPermutation(handle, nnzV, Pa);
	cusparseSafeCall(cusparseXcoosortByRow(handle, NrowsA, NcolsA,
		nnzV,
		d_cooRowIndA,
		d_A_ColIndices,
		Pa,
		pBuffera));

	cusparseSafeCall(cusparseDgthr(handle, nnzV, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzV,
		NrowsA,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));

		
		
   size_t pivot_dimensionsrow[1] = {nnzR};
   size_t pivot_dimensionscolumn[1] = {NcolsA+1}; 
   size_t pivot_dimensionsvalue[1] = {nnzV};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORT = (double *)mxGPUGetData(VAL_SORT1);

	cusparseSafeCall(cusparseDcsr2csc(handle, NrowsA, NcolsA, nnzV, d_A, d_A_RowIndices, d_A_ColIndices, VALSORT, ROWSORT, COLSORT, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE));

	//gpuErrchk(cudaFree(d_A));
	//gpuErrchk(cudaFree(d_A_RowIndices));
	//gpuErrchk(cudaFree(d_A_ColIndices));
	//gpuErrchk(cudaFree(d_cooRowIndA));
	
	mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(ROW_A);
    mxGPUDestroyGPUArray(COL_A);
    mxGPUDestroyGPUArray(COO_A);
	
	
	
	gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
    
    
   mwSize nnzm=(mwSize)nnzV;
   OUTPUTMATRIX = mxCreateSparse(NrowsA,NcolsA,nnzm,mxREAL);
   
    
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  

        irs = static_cast<mwIndex *> (mxMalloc (nnzR * sizeof(mwIndex)));
       #pragma omp parallel for shared(nnzR) private(i)
         for ( i = 0; i < nnzR; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i])-1; 
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((NcolsA+1) * sizeof(mwIndex)));
        int nc1= NcolsA+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i])-1;
            }
             
        mxDouble* PRS = (mxDouble*) mxMalloc (nnzV * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnzV * sizeof(mxDouble), cudaMemcpyDeviceToHost));

           
    
        
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
        mxSetNzmax(OUTPUTMATRIX, (static_cast<mwSize>(NNZMAXA)));
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);

 
      mxDestroyArray(ROWx);
      mxDestroyArray(COLUMNx);
      mxDestroyArray(VALUEx);
      
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
	  
      cusparseDestroyMatDescr(descrA);
      cusparseDestroy(handle);

               }
           
               
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
