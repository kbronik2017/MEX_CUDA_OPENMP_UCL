[![GitHub issues](https://img.shields.io/github/issues/kbronik2017/MEX_CUDA_OPENMP_UCL)](https://github.com/kbronik2017/MEX_CUDA_OPENMP_UCL/issues)
[![GitHub forks](https://img.shields.io/github/forks/kbronik2017/MEX_CUDA_OPENMP_UCL)](https://github.com/kbronik2017/MEX_CUDA_OPENMP_UCL/network)
[![GitHub stars](https://img.shields.io/github/stars/kbronik2017/MEX_CUDA_OPENMP_UCL)](https://github.com/kbronik2017/MEX_CUDA_OPENMP_UCL/stargazers)



# CUDA-Cusparse-OpenMp Matlab (Sparse matrix manipulation)

 * The CUDA-Cusparse code can handle/work with  any type of the input mxArrays, GPUarray or standard matlab CPU array as input {prhs[0], ...  := mxGPUArray or CPU Array}[double or complex double]
 * Create sparse matrix  
 * Z=CuMatlab_sparse(X) 
 * Z=CuMatlab_sparse(X,Y)
 * Z=CuMatlab_sparse(X,Y,Z)
 * Z=CuMatlab_sparse(X,Y,Z,row,column) 
 * Z=CuMatlab_sparse(X,Y,Z,row,column,nz)
 * etc
 
 # CUDA Matlab (Dense matrix manipulation)
 
 * The CUDA code can handle/work with  any type of the input mxArrays,  GPUarray or standard matlab CPU array as input {prhs[0], ... := mxGPUArray or CPU Array}
 * C=MM3D_CUBLAS(A,B,alpha),  C=A*B*alpha
 * etc

 
