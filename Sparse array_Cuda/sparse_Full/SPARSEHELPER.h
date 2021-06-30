/*
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */
#if !defined(SPARSEHELPER_H_)
#define SPARSEHELPER_H_

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cusparse_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <omp.h>


   struct MATRIX
{
	int row_C;
	int column_C;
	double value_C;
	bool checked=false;

	MATRIX() : row_C(0), column_C(0), value_C(0.0) {}
	MATRIX(int row, int  column, double value) : row_C(row), column_C(column), value_C(value) {}

	bool operator < (const MATRIX &input) const
	{
		return row_C < input.row_C;
	}
}; 

   struct MATRIXF
{
	int row_C;
	int column_C;
	float value_C;
	bool checked=false;

	MATRIXF() : row_C(0), column_C(0), value_C(0.f) {}
	MATRIXF(int row, int  column, float value) : row_C(row), column_C(column), value_C(value) {}

	bool operator < (const MATRIXF &input) const
	{
		return row_C < input.row_C;
	}
};

   struct MATRIXC
{
	int row_C;
	int column_C;
	double value_C_real;
    double value_C_img;
	bool checked=false;

	MATRIXC() : row_C(0), column_C(0), value_C_real(0), value_C_img(0) {}
	MATRIXC(int row, int  column, double valreal, double valimg ) : row_C(row), column_C(column), value_C_real(valreal),value_C_img(valimg){}

	bool operator < (const MATRIXC &input) const
	{
		return row_C < input.row_C;
	}
};

bool max_elem(int i, int j) {
return i<j;
} 

   
 
double  *Jc_Set(mxArray const *ptr)
{

    mwIndex  *jc ;
    double *out=0;
    int total=0;
    int      col;
    mwIndex   jc1, jc2;
    int  n;
    jc = mxGetJc(ptr);

    n = static_cast<int> (mxGetN(ptr));
        int temp;
        int i;
out = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
 // #pragma omp parallel for shared(n) private(col)
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     out[total+i] = static_cast<double> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

return out;
}

void Jc_SetDX(mxArray const *ptr, double JCD[])
{

    mwIndex  *jc ;
    //double *out=0;
    int total=0;
    int      col;
    mwIndex   jc1, jc2;
    int  n;
    jc = mxGetJc(ptr);
    n = static_cast<int> (mxGetN(ptr));
   
        int temp;
        int i;
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     JCD[total+i] = static_cast<double> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }


}



  int *Ir_Data (mxArray const *ptr)
{
  mwIndex *Irx = mxGetIr(ptr);
   int  nz = static_cast<int> (mxGetNzmax(ptr));
  
 // int  val=omp_get_max_threads();
 //  omp_set_num_threads(val);
  int *Ir = 0;
  int i;
      Ir = static_cast<int *> (mxMalloc (mxGetNzmax(ptr) * sizeof(int)));
      //#pragma acc loop gang(16), vector(32)
      #pragma omp parallel for shared(nz) private(i)
      for (i=0 ; i < nz; i++)
        {

            Ir[i] = static_cast<int> (Irx[i]);
        }
    
  return Ir;
}



   
  double *Ir_DataD (mxArray const *ptr)
{
  mwIndex *Irx = mxGetIr(ptr);
  
   int  nz = static_cast<int> (mxGetNzmax(ptr));
  // int  val=omp_get_max_threads();
  // omp_set_num_threads(val);  
  double *Ir = 0;
  int i;
      Ir = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(i)
      for ( i = 0; i < nz; i++)
        {
            Ir[i] = static_cast<double> (Irx[i]);
        }
    
  return Ir;
}  

  void Ir_DataDX (mxArray const *ptr, double IRD[])
{
  mwIndex *Irx = mxGetIr(ptr);
  
   int  nz = static_cast<int> (mxGetNzmax(ptr));
  // int  val=omp_get_max_threads();
 //  omp_set_num_threads(val);
  int i;
       #pragma omp parallel for shared(nz) private(i)
      for ( i = 0; i < nz; i++)
        {

            IRD[i] = static_cast<double> (Irx[i])+1;
        }
 
}  


int *Jc_Data (mxArray const *ptr)
{
  mwIndex *Jcx = mxGetJc(ptr);
 int   col;
  mwSize columns;
  //int  val=omp_get_max_threads();
  // omp_set_num_threads(val);
  columns = mxGetN(ptr);
 
  int *Jc = 0;
       
      Jc = static_cast<int *> (mxMalloc ((columns+1) * sizeof(int)));
	  int nz=(static_cast<int> (columns))+1;
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(col)
      for (col = 0; col < nz; col++)
        {
            Jc[col] = static_cast<int> (Jcx[col]);
        }
    
  return Jc;
} 


  double * Ir_DataDXY (mxArray const *ptr)
{
  mwIndex *Irx = mxGetIr(ptr);
 
  int  nz = static_cast<int> (mxGetNzmax(ptr));
  // int  val=omp_get_max_threads();
  // omp_set_num_threads(val);
  double *Ir = 0;
  int i;
      Ir = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(i)
      for (i = 0; i < nz; i++)
        {

            Ir[i] = static_cast<double> (Irx[i])+1;
        }
    
  return Ir;
} 
  
  
double * Jc_SetDXY(mxArray const *ptr)
{

    mwIndex  *jc ;
    double *out=0;
    int total=0;
    int     col;
    mwIndex   jc1, jc2;
    
    jc = mxGetJc(ptr);

    
	int  n = static_cast<int> (mxGetN(ptr));
        int temp;
        int i;
out = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     out[total+i] = static_cast<double> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

return out;
}

 int  *Jc_SetInt(mxArray const *ptr)
{

    mwIndex  *jc ;
    int *out=0;
    int total=0;
    int      col;
    mwIndex   jc1, jc2;
    
    jc = mxGetJc(ptr);
int  n = static_cast<int> (mxGetN(ptr));
    
        int temp;
        int i;
out = static_cast<int *> (mxMalloc (mxGetNzmax(ptr) * sizeof(int)));
 #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     out[total+i] = static_cast<int> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

return out;
}
 
  void SetIr_Data (mxArray const *ptr, int input[])
{
  mwIndex *Irx = mxGetIr(ptr);
 
  int  nz = static_cast<int> (mxGetNzmax(ptr));
  //int  val=omp_get_max_threads();
  // omp_set_num_threads(val);
  int i;
       #pragma omp parallel for shared(nz) private(i)
      for (i = 0; i < nz; i++)
        {
            input[i] = static_cast<int> (Irx[i]) +1;
        }
 
}
  
  
 void SetJc_Int(mxArray const *ptr, int input [])
{

    mwIndex  *jc ;
    //int *out=0;
    int total=0;
    int      col;

    mwIndex   jc1, jc2;
    
    jc = mxGetJc(ptr);
int  n = static_cast<int> (mxGetN(ptr));
    
        int temp;
        int i;
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     input[total+i] = static_cast<int> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

}
   void Ir_DataGetSet (mxArray const *ptr, int input[], int nz)
{
  mwIndex *Irx = mxGetIr(ptr);

int i;
     #pragma omp parallel for shared(nz) private(i)
      for ( i = 0; i < nz; i++)
        {

            input[i] = static_cast<int> (Irx[i]);
        }
      
}
 void Jc_DataGetSet (mxArray const *ptr, int input[],int columns)
{
  mwIndex *Jcx = mxGetJc(ptr);
 
int end=columns+1;
int col;


      #pragma omp parallel for shared(end) private(col)

      for ( col = 0; col < end; col++)
        {

            input[col] = static_cast<int> (Jcx[col]);
        }
      
} 
   void Ir_DataGetSetDXY (mxArray const *ptr, double input[] , int nz)
{
  mwIndex *Irx = mxGetIr(ptr);

  int i;
     // Ir = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(i)
      for (i = 0; i < nz; i++)
        {

            input[i] = static_cast<double> (Irx[i])+1;
        }
    
  
}
void  Jc_GetSetDXY(mxArray const *ptr, double input[])
{

    mwIndex  *jc ;
    
    int total=0;
    int     col;
    mwIndex   jc1, jc2;
    
    jc = mxGetJc(ptr);

    
int  n = static_cast<int> (mxGetN(ptr));	
        int temp;
        int i;
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     input[total+i] = static_cast<double> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

}   
     void value_DataGetSetDXY (mxArray const *ptr, double input[], int nz)
{
  mxDouble *Vlx = mxGetDoubles(ptr);

  int i;
     // Ir = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(i)
      for (i = 0; i < nz; i++)
        {

            input[i] = static_cast<double> (Vlx[i]);
        }
    
  
} 
   void Ir_DataGetSetIXY (mxArray const *ptr, int input[] , int nz)
{
  mwIndex *Irx = mxGetIr(ptr);

  int i;
     // Ir = static_cast<double *> (mxMalloc (mxGetNzmax(ptr) * sizeof(double)));
      //#pragma acc loop gang(16), vector(32)
       #pragma omp parallel for shared(nz) private(i)
      for (i = 0; i < nz; i++)
        {

            input[i] = static_cast<int> (Irx[i])+1;
        }
    
  
}
void  Jc_GetSetIXY(mxArray const *ptr, int input[])
{

    mwIndex  *jc ;
    
    int total=0;
    int     col;
    mwIndex   jc1, jc2;
    
    jc = mxGetJc(ptr);

    
int  n = static_cast<int> (mxGetN(ptr));	
        int temp;
        int i;
  #pragma omp parallel for shared(n) private(col)
        for (col=0; col<n; col++)  {
            jc1 = jc[col];
            jc2 = jc[col+1];
            temp=static_cast<int> (jc[col+1]-jc[col]);
            
            total=static_cast<int> (jc[col]-jc[0]);
            
            if (jc1 == jc2)
                continue;
            else {
    //#pragma omp parallel for shared(temp) private(i)
                for ( i=0; i < temp; i++)  {
                    
                     input[total+i] = static_cast<int> (col+1);  
                      //out[col]=col+1;
                      //total++
                    
                }
            }
        }

}     
#endif /* !defined(SPARSEHELPER_H_) */
