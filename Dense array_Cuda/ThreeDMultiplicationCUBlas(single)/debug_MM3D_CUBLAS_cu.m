
 function debug_MM3D_CUBLAS_cu(bDebug)

% /*
%  * Three dimensional Matrix Multiplication using cublas
%  * 
%  * This CUDA code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0], prhs[1] := mxGPUArray or CPU Array}
%  * gpuArray output, C=MM3D_CUBLAS(A,B,alpha) C=A*B*alpha.
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
delete MM3D_CUBLAS.mex*
 
 
    if(bDebug)
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver MM3D_CUBLAS.cu  3DMultiplicationCUBlas.cu
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  CuMatlab_sparse.cu  Cu_sparseO.cu  Cu_sparseZ.cu  Cu_sparseD.cu Cu_sparseDX.cu Cu_sparseDY.cu Cu_sparseSX.cu Cu_sparseSY.cu
    mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver  MM3D_CUBLAS.cu  3DMultiplicationCUBlas.cu
    end
 
   
    %  
    
    X = single(rand(5,3,4));
    Y = single(rand(3,7,4));
%     Matlab=X*Y   Matlab does not support 3d Matrix Multiplication unless
    Matlab(:,:,1)=X(:,:,1)*Y(:,:,1);
    Matlab(:,:,2)=X(:,:,2)*Y(:,:,2);
    Matlab(:,:,3)=X(:,:,3)*Y(:,:,3);
    Matlab(:,:,4)=X(:,:,4)*Y(:,:,4);
    
    Matlab
    Cuda=MM3D_CUBLAS (X, Y, 1)   
    
    disp('finished without error');
    clear mmc;
 