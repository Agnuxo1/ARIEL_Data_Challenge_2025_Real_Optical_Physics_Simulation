@echo off
REM Compile ARIEL Kaggle Standalone Submission (CPU-only, optimized)

setlocal

set "VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

call "%VS_VCVARS%"
if errorlevel 1 goto :error

set "INCLUDE_DIRS=/I"%CD%" /I"E:\eigen-3.4.0" /I"C:\cnpy" /I"E:\openblas_install\include" /I"C:\zlib_install\include""

echo ========================================
echo ARIEL KAGGLE COMPILATION
echo ========================================
echo.

echo Cleaning previous builds...
if exist ariel_kaggle_standalone.obj del /f /q ariel_kaggle_standalone.obj
if exist ariel_kaggle_standalone.exe del /f /q ariel_kaggle_standalone.exe

echo.
echo Compiling ariel_kaggle_standalone.cpp (CPU-optimized)...
cl /std:c++17 /O2 /EHsc /MD %INCLUDE_DIRS% /DNDEBUG /c ariel_kaggle_standalone.cpp
if errorlevel 1 goto :error

echo.
echo Linking ariel_kaggle_standalone.exe (Kaggle-ready)...
cl /Fe:ariel_kaggle_standalone.exe ariel_kaggle_standalone.obj ^
    "C:\cnpy\Release\cnpy.lib" ^
    "E:\openblas_install\lib\libopenblas.lib" ^
    "C:\zlib_install\lib\z.lib" ^
    legacy_stdio_definitions.lib
if errorlevel 1 goto :error

echo.
echo ========================================
echo KAGGLE BUILD SUCCESSFUL!
echo ========================================
echo Executable: ariel_kaggle_standalone.exe
echo Ready for Kaggle submission package
echo ========================================
goto :eof

:error
echo ========================================
echo KAGGLE BUILD FAILED!
echo ========================================
exit /b 1

endlocal