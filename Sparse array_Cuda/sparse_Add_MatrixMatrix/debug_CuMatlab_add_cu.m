
 function debug_CuMatlab_add_cu(bDebug)
% /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
%  * Sparse/Dense matrix-sparse/dense matrix addition   Z=CuMatlab_add(Sparse/Dense(X),Sparse/Dense(Y), alpha, beta).
%  * Z= alpha*X+beta*Y
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
delete CuMatlab_add.mex*
    if(bDebug)
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_sparseMA.cu Cu_MA.cu
  mmc -g  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver CuMatlab_add.cu /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_sparseMA.cu Cu_MA.cu
  mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  -lcusolver CuMatlab_add.cu /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    end
 
   
    %  
    Y = magic(4);
    X = magic(4);
    Y=sparse(Y);
    alpha=8;
    beta=-3;
    % 
    Z=CuMatlab_add(Y, X, alpha, beta)
    
    Zm=alpha*Y+beta*X;
    Zm=sparse(Zm)
    verify= Zm-Z
 
 
    disp('finished without error');
    clear mmc;
 