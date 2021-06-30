
 function debug_CuMatlab_svd_cu(bDebug)
% /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double or complex double]
%  * This  computes the singular value decomposition (SVD) of sparse/dense matrix  [U, S, V]=CuMatlab_svd(Sparse/Dense(X))
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
delete CuMatlab_svd.mex*
 

    if(bDebug)
%       mexcuda -g -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_find.cu Cu_find.cu 
      mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver CuMatlab_svd.cu  /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_find.cu Cu_find.cu 
   mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  -lcusolver  CuMatlab_svd.cu   /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    
    end
 
   
    %  
    X = magic(4);
    
    [Ux, Sx, Vx]=CuMatlab_svd(X)
    
    A = [1 2; 3 4; 5 6; 7 8]+3i*[1 2; 3 4; 5 6; 7 8];
    A=gpuArray(A); % 
    
    [Uy, Sy, Vy]=CuMatlab_svd(A)
    
    

    disp('finished without error.');
    clear mmc;
 