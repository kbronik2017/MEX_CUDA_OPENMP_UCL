
 function debug_ELA_CUDA_cu(bDebug)

% /*
%  * ElementwiseAddition
%  * 
%  * This CUDA code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0], prhs[1] := mxGPUArray or CPU Array}
%  * gpuArray output, C=ELA_CUDA(A,B, alpha, beta) C=A*alpha+B*beta.
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
delete ELA_CUDA.mex*
 
 
    if(bDebug)
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver ELA_CUDA.cu  ElementwiseAddition.cu 
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver  ELA_CUDA.cu  ElementwiseAddition.cu
    end
 
   
    % 
    A=single(rand(8,5));
    B=single(rand(8,5));
    Matlab2D=A+B
    Cuda2D=ELA_CUDA(A, B, 1, 1)
    
    Y = single(rand(2,3,4));
    X= single(rand(2,3,4));
    Matlab3D=X+Y
    Cuda3D=ELA_CUDA_3D (X, Y, 1, 1)   
    
    disp('finished without error');
    clear mmc;
 