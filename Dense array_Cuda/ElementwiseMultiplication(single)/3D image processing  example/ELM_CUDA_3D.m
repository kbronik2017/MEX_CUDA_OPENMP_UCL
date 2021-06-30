%
 % Three dimensional Element wise Multiplication
 % 
 % This MATLAb-CUDA  Wrapper function can handle/work with  any type of the input mxArrays, 
 % GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 % gpuArray output, C=ELM_CUDA_3D(A,B).
 % Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 % Wellcome Trust Centre for Neuroimaging
 % Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 % Copyright 2018
 % Kevin Bronik
 %/
 
 function OUTCoefs = ELM_CUDA_3D(data1,data2)


[ex1,wy1,zy1]=size(data1);
[ex2,wy2,zy2]=size(data2);
if (ex1~=ex2)|| (wy1~=wy2)|| (zy1~=zy2)
  error('Array dimensions must match.');  
end


if  isa(data1,'gpuArray') && isa(data2,'gpuArray')
    

OUTCoefs=zeros(ex1,wy1,zy1, 'gpuArray');

for i=1:zy1
    tempData1=gpuArray(data1(:,:,i));
    tempData2=gpuArray(data2(:,:,i));
    Coefs=ELM_CUDA(tempData1,tempData2);
       
     OUTCoefs(:,:,i)=Coefs;
end


elseif ~isa(data1,'gpuArray') && ~isa(data2,'gpuArray')
OUTCoefs=zeros(ex1,wy1,zy1);

for i=1:zy1
    tempData1=data1(:,:,i);
    tempData2=data2(:,:,i);
    Coefs=ELM_CUDA(tempData1,tempData2);
       
     OUTCoefs(:,:,i)=Coefs;
end
    
else
    error('Input(FIRST and SECOND ARGUMENTS) must be array, or gpuArray object, not  %s.',data1,data2);

end


 end %function


 