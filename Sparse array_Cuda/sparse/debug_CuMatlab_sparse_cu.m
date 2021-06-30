
 function debug_CuMatlab_sparse_cu(bDebug)

%  /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0],prhs[1],prhs[2]  := mxGPUArray or CPU Array}[double or complex double]
%  * Create sparse matrix  
%  * Z=CuMatlab_sparse(X) 
%  * Z=CuMatlab_sparse(X,Y)
%  * Z=CuMatlab_sparse(X,Y,Z)
%  * Z=CuMatlab_sparse(X,Y,Z,row,column) 
%  * Z=CuMatlab_sparse(X,Y,Z,row,column,nz)
%  * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
%  * Wellcome Trust Centre for Neuroimaging
%  * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
%  * Copyright 2018
%  * Kevin Bronik
%  */
 
 if ismac
    % Code to run on Mac plaform
elseif isunix
    % checks

if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-9.0 /usr/local/cuda"')
end
end

newpath = fileparts(mfilename('fullpath'));
cd(newpath);
delete CuMatlab_sparse.mex*
 
 
    if(bDebug)
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver CuMatlab_sparse.cu  /NODEFAULTLIB:vcomp.lib libiomp5md.lib 
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver  CuMatlab_sparse.cu   /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    end
 
   
    %  
    Y = magic(4);
    Zmc=sparse(Y)
    Zcc=CuMatlab_sparse(Y)
    verify= Zmc-Zcc 
    
    Y=gpuArray(Y); % ) 
    Zm=sparse(Y)
    Zc=CuMatlab_sparse(Y)
    verifygp= Zm-Zc 
    
    disp('finished without error');
    clear mmc;
 