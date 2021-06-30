function varargout = mmc(varargin)
%MEXCUDA   Compile MEX-function for GPU computation
%
%   Usage:
%       mexcuda [options ...] file [files ...]
%
%   Description:
%       MEXCUDA compiles and links source files into a shared library
%       called a MEX-file, executable from within MATLAB. MEXCUDA is an
%       extension of MATLAB's MEX function. It compiles MEX-files written
%       using the CUDA C++ framework with NVIDIA's nvcc compiler, allowing
%       the files to define and launch GPU kernels. In addition, it exposes
%       the GPU MEX API to allow the MEX-file to read and write gpuArrays.
%
%   MEXCUDA provides the following command line options in addition to those
%   provide by MEX:
%       -dynamic
%           Dynamic parallelism: compiles MEX-files that define kernels
%           that launch other kernels.
%       -G
%           Generate debug information for device code. This makes it
%           possible to step through kernel code line by line in one of
%           NVIDIA's debugging applications (NSight or cuda-gdb). To enable
%           debugging of host code use -g.
%
%   Command line options not available to MEXCUDA:
%       -compatibleArrayDims
%           Use of -R2018a is implicit for MEXCUDA and cannot be
%           overridden.
%
%   For more information, see
%           <a href="matlab:helpview(fullfile(docroot,'distcomp','distcomp.map'),'EXECUTE_MEX_CUDA')">Run MEX functions including CUDA code</a>
%
%   See also MEX

% Copyright 2015 The MathWorks, Inc.

% Illegal MEX options
if any(strcmp(varargin, '-compatibleArrayDims'))
    error(message('parallel:gpu:mex:MEXCUDACompatibleArrayDimsNotSupported'));
end

% -setup diverts straight to MEX
if any(strcmp(varargin, '-setup'))
    [varargout{1:nargout}] = mex(varargin{:});
    return;
end

% Detect verbose mode
verbose = false;
if any(strcmp(varargin, '-v'))
    verbose = true;
end

% Process new options
[useDynamic, varargin] = iProcessOptionFlag(varargin, '-dynamic');
[useDeviceDebug, varargin] = iProcessOptionFlag(varargin, '-G');

% If the user has specified a custom options file, pass that directly on to
% MEX. Otherwise determine options file from settings or choose a default.
chosenOptions = '';
if ~any(strcmp(varargin, '-f'))

    % Load the C++ compiler preference.
    compilerForMex = mex.getCompilerConfigurations('C++','selected');
    hostName = '';
    if ~isempty(compilerForMex)
        hostName = compilerForMex(1).ShortName;
    end

    % Get options files and order by Priority
    arch = computer('arch');
    basepath = fullfile(matlabroot, 'toolbox', 'distcomp', 'gpu', 'extern', 'src', 'mex');
    filelist = iGetOptionsFiles(useDynamic, fullfile(basepath, arch));
    priorities = iGetAttributes(filelist, 'Priority');
    [~, I] = sort(priorities);
    filelist = filelist(I);

    % If the user has selected a host compiler and it is supported, make
    % sure it is the first one checked.
    if ~isempty(hostName)
        hostCompilers = iGetAttributes(filelist, 'HostCompiler');
        selected = strcmpi(hostName, hostCompilers);
        if any(selected)
            % Move to top of list
            filelist = [filelist(selected); filelist(~selected)];
        else
            % Let user know that a different compiler will be used
            warning(message('parallel:gpu:mex:MEXCUDAWarnSelectedCompilerNotUsed'));
        end
    end

    % Try to use each option file in turn to compile the example code, just to
    % test whether the compiler is present
    %  Suppress MEX warnings while we're doing this
    warnState = warning('QUERY', 'Matlab:mex:Warn');
    warning('OFF', 'Matlab:mex:Warn');
    exampleFile = fullfile(basepath, 'mexGPUExample.cu'); %#ok<NASGU>
    chosenOptions = filelist{1};
    success = false;
    i = 1;
    while ~success && i <= length(filelist)
        try
            if verbose
                m = message('parallel:gpu:mex:MEXCUDAVerboseOptsPreamble', filelist{i});
                fprintf('%s', m.getString);
            end
            % Run mex in 'no execute' mode (-n). Suppress output using evalc.
            evalc('mex(''-R2018a'', ''-silent'', ''-n'', ''-f'', filelist{i}, exampleFile)');
            % Success if we got here
            if verbose
                m = message('parallel:gpu:mex:MEXCUDAVerboseOptsSuccess');
                fprintf('%s\n', m.getString);
            end
            chosenOptions = filelist{i};
            success = true;
        catch
            % Ignore error at this point, just try next options file
            if verbose
                m = message('parallel:gpu:mex:MEXCUDAVerboseOptsFailed');
                fprintf('%s\n', m.getString);
            end
        end
        i = i + 1;
    end
    %  Turn MEX warnings back on
    warning(warnState);

    if verbose && ~success
        warning(message('parallel:gpu:mex:MEXCUDAVerboseFindOptsFailure'));
    end
end

% Set user flags to NVCC
nvccFlags = '-Xcompiler "/MD /openmp"';

% nvcc -ccbin=mpic++ cuda-mpi.cu
if useDeviceDebug
    nvccFlags = [nvccFlags '-G '];
end

% Construct mex arguments
mexArguments = { '-R2018a' };
if chosenOptions
    mexArguments = [ mexArguments, '-f', chosenOptions ];
end
mexArguments = [ mexArguments, ...
                 ['NVCC_FLAGS="' nvccFlags  '"'], ...
                 varargin{:} ];

% Call mex
if verbose
    % Interleave with spaces for pretty print
    nArgs = length(mexArguments);
    printedArgs = [mexArguments; repmat({' '}, 1, nArgs)];
    disp(['mex ', printedArgs{:}]);
end
try
    [varargout{1:nargout}] = mex(mexArguments{:});
catch ME
    rethrow(ME);
end

end

function [flagSet, options] = iProcessOptionFlag(options, flag)
% Retrieve an option to CUDAMEX that cannot be passed on to MEX

flagSet = false;
mask = strcmp(options, flag);
if any(mask)
    flagSet = true;
    % Remove flag from the set of options
    options(mask) = [];
end
end

function filelist = iGetOptionsFiles(useDynamic, basepath)
% Find all options files regardless of directory structure, and then
% separate the standard ones from those for dynamic parallelism.
matchString = 'nvcc_*.xml';
dynamicSuffix = '_dynamic.xml';

% Recursively search child directories
filelist = {};
allFiles = dir(basepath);
for i = 1:length(allFiles)
    thisFile = allFiles(i);
    if thisFile.isdir && thisFile.name(1) ~= '.'
        filelist = [filelist; iGetOptionsFiles(useDynamic, fullfile(basepath, thisFile.name))]; %#ok<AGROW>
    end
end

% Get a list of all the valid options filenames in this directory
searchPath = fullfile(basepath, matchString);
filelistAsStruct = dir(searchPath);
filenamesAsCell = {filelistAsStruct.name}';

% Choose files with or without the '_dynamic' suffix, depending on the
% useDynamic flag.
dynamicOptions = cellfun(@(s) ~isempty(s), regexp(filenamesAsCell, [dynamicSuffix '$']));
filenamesAsCell = filenamesAsCell(dynamicOptions == useDynamic);

% Convert to full paths and add to the filelist
addFiles = cellfun(@(f) fullfile(basepath, f), filenamesAsCell, 'UniformOutput', false);
filelist = [filelist; addFiles];
end

function attributes = iGetAttributes(optionsFiles, attribName)
% Get a cell array of attributes from the MEX options XML files. This
% assumes the attribute is identified by ATTRIBNAME="VALUE" somewhere on a
% line of the file, and does not check for proper XML syntax, so it cannot
% be used generically.

n = length(optionsFiles);
attributes = cell(n, 1);
for i = 1:n
    thisFile = optionsFiles{i};
    fid = fopen(thisFile);
    found = false;
    while ~feof(fid)
        thisLine = fgetl(fid);
        attrib = regexp(thisLine, ['\s' attribName '="(.)*"'], 'tokens');
        if ~isempty(attrib)
            attributes(i) = attrib{1}(1);
            found = true;
            break;
        end
    end
    % If the attribute isn't there, make sure it will end up at the bottom
    % of a sorted list
    if ~found
        attributes{i} = 'Z';
    end
    fclose(fid);
end
end
