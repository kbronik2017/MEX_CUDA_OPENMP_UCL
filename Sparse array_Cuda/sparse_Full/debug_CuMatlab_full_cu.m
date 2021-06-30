
 function debug_CuMatlab_full_cu(bDebug)
% /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double/complex double]
%  * Sparse/Dense --> Dense,   Z=CuMatlab_full(Sparse/Dense(X)).
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
delete CuMatlab_full.mex*
    if(bDebug)
%         mexcuda -g -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_full.cu Cu_full.cu 
      mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver  CuMatlab_full.cu  /NODEFAULTLIB:vcomp.lib libiomp5md.lib
  
  
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_full.cu Cu_full.cu 
      mmc    '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  -lcusolver  CuMatlab_full.cu  /NODEFAULTLIB:vcomp.lib libiomp5md.lib
  
    end
 
%    CFLAGS="$CFLAGS -fopenmp"
% COMPFLAGS='$COMPFLAGS /openmp'
    %  
    Y = magic(4);
     %  
    Z=sparse(Y);
    Z=gpuArray(Z);
    M=full(Z)
    R=CuMatlab_full(Z)
 
    disp('finished without error.');
    clear mmc;
 