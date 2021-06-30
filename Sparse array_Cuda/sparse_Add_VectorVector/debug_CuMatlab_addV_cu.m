
 function debug_CuMatlab_addV_cu(bDebug)

% /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
%  * Sparse/Dense vector-sparse/dense vector addition   Z=CuMatlab_addV(Sparse/Dense(X),Sparse/Dense(Y), alpha).
%  * Z= alpha*X+Y
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
delete CuMatlab_addV.mex*
    if(bDebug)
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_sparseMA.cu Cu_MA.cu
  mmc -g  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver CuMatlab_addV.cu /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_sparseMA.cu Cu_MA.cu
  mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver  CuMatlab_addV.cu /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    end
 
    E = [1;  0;  -4;  1; 0; -2;  19; 3; 0; -2;  19;  3; 1;  0];
    F = [0; -2;  19;  3; 1;  0;  -4; 1; -4;  1; 0; -2;  19; 3];

   
    Z1=3*E+F;
    % 
    Z2=CuMatlab_addV(E, F, 3);
    
   
    
    verifyEF= Z1-Z2
    %  
    Y = [1; 0;  -4; 1; 0; -2;  19; 3];
    X = [0; -2;  19; 3; 1; 0;  -4; 1];
    X=gpuArray(X);
    Y=gpuArray(Y);
    alpha= -8;
    
    % 
    Z=CuMatlab_addV(Y, X, alpha)
    
    Zm=alpha*Y+X
    
    verifyXY= Zm-Z
 
 
    disp('finished without error');
    clear mmc;
 