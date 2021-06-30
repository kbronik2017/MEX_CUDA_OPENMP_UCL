
 function debug_CuMatlab_find_cu(bDebug)
% /*
%  * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double or complex double]
%  * Create row, column, value vectors from sparse/dense matrix  [row, column, value]=CuMatlab_find(Sparse/Dense(X))
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
delete CuMatlab_find.mex*
 

    if(bDebug)
%       mexcuda -g -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_find.cu Cu_find.cu 
      mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static -lcusolver CuMatlab_find.cu  /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    
    else
%       mexcuda -largeArrayDims   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device -lcusparse -lcudart_static  CuMatlab_find.cu Cu_find.cu 
   mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include' -lcusparse -lcudart -lcublas -lcublas_device  -lcudart_static  -lcusolver  CuMatlab_find.cu   /NODEFAULTLIB:vcomp.lib libiomp5md.lib
    
    end
 
   
    %  
    Y = magic(4);
    [row, column, value]=find(Y);
    [rowcu, columncu, valuecu]=CuMatlab_find(Y);
    [row, column, value]
    [rowcu, columncu, valuecu]
    Y=gpuArray(Y); % 
    [rowm, columnm, valuem]=find(Y);
    [rowc, columnc, valuec]=CuMatlab_find(Y);
    [rowc, columnc, valuec]
    [rowm, columnm, valuem]

    disp('finished without error.');
    clear mmc;
 