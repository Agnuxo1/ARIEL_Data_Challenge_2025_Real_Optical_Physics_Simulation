@echo off
REM Compile the ARIEL hybrid model in CPU-only mode (sin CUDA)

setlocal

set "VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

call "%VS_VCVARS%"
if errorlevel 1 goto :error

set "INCLUDE_DIRS=/I"%CD%" /I"E:\eigen-3.4.0" /I"C:\cnpy" /I"E:\openblas_install\include" /I"C:\zlib_install\include""

echo Limpiando objetos previos...
if exist ariel_trainer.obj del /f /q ariel_trainer.obj

echo Compilando ariel_trainer.cpp (CPU-only)...
cl /std:c++17 /O2 /EHsc /MD %INCLUDE_DIRS% /c ariel_trainer.cpp
if errorlevel 1 goto :error

echo Enlazando ariel_trainer.exe (CPU-only)...
cl /Fe:ariel_trainer.exe ariel_trainer.obj ^
    "C:\cnpy\Release\cnpy.lib" ^
    "E:\openblas_install\lib\libopenblas.lib" ^
    "C:\zlib_install\lib\z.lib" ^
    legacy_stdio_definitions.lib
if errorlevel 1 goto :error

echo Compilación CPU completada: ariel_trainer.exe
goto :eof

:error
echo Error en la compilación CPU!
exit /b 1

endlocal


