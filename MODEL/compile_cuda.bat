@echo off
REM Compile and link the ARIEL hybrid model using MSVC + CUDA

setlocal

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

call "%VS_VCVARS%"
if errorlevel 1 goto :error

set "CL_INCLUDE_DIRS=/I"%CD%" /I"E:\eigen-3.4.0" /I"C:\cnpy" /I"E:\openblas_install\include" /I"C:\zlib_install\include" /I"%CUDA_PATH%\include""
set "NVCC_INCLUDE_DIRS=-I"%CD%" -I"E:\eigen-3.4.0" -I"C:\cnpy" -I"E:\openblas_install\include" -I"C:\zlib_install\include" -I"%CUDA_PATH%\include""

echo Compiling ariel_trainer.cpp with cl (/MD) [CUDA build]...
cl /std:c++17 /O2 /EHsc /MD %CL_INCLUDE_DIRS% /DARIEL_USE_CUDA /c ariel_trainer.cpp
if errorlevel 1 goto :error

echo Compiling nebula_kernels.cu with nvcc...
"%CUDA_PATH%\bin\nvcc.exe" -c nebula_kernels.cu -o nebula_kernels.obj -Xcompiler "/EHsc /std:c++17 /O2 /MD /DARIEL_USE_CUDA" -arch=sm_70 %NVCC_INCLUDE_DIRS%

if errorlevel 1 goto :error

echo Linking ariel_trainer.exe (CUDA)...
cl /Fe:ariel_trainer.exe ariel_trainer.obj nebula_kernels.obj ^
    "C:\cnpy\Release\cnpy.lib" ^
    "E:\openblas_install\lib\libopenblas.lib" ^
    "C:\zlib_install\lib\z.lib" ^
    legacy_stdio_definitions.lib ^
    /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cufft.lib cublas.lib
if errorlevel 1 goto :error

echo Build successful: ariel_trainer.exe
goto :eof

:error
echo Build failed!
exit /b 1

endlocal

