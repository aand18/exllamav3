@echo off
rem Build the spectre branch (v1.6.4 + arms) as a Windows wheel, in a NEW tree.
rem
rem Recipe is the one that already worked on this box (Downloads\exl3-build-154\_a15_build.bat),
rem kept verbatim where it matters:
rem   - CUDA v13.3, NOT v13.2: the cu132 in the 1.5.4 wheel name is nominal, the toolkit on PATH
rem     is 13.3 and torch is cu130. Do not go hunting for a 13.2 install.
rem   - MSVC + Windows SDK set BY HAND (VsDevCmd does not propagate reliably to child shells here).
rem   - TORCH_CUDA_ARCH_LIST=8.9 (4090), MAX_JOBS=4, CUDAHOSTCXX=cl.exe, DISTUTILS_USE_SDK=1.
rem   - no ninja in the venv -> setuptools uses the distutils fallback, 190 TUs, ~33 min.
rem   - the venv has NO pip and NO wheel, so this uses setup.py directly, never `pip wheel`.
rem     setuptools 78 ships bdist_wheel, so `setup.py bdist_wheel` gives us a real wheel.
rem
rem Nothing is installed: this only writes dist\*.whl. The v1.4.9 clone at
rem C:\Users\yoho\Downloads\exllamav3 is never touched.
setlocal
set SCRATCH=C:\Users\yoho\Downloads\exllamav3-spectre
set L=%SCRATCH%\build_wheel.log
set VSPATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set SDK=10.0.26100.0
set PYTHON=C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe

if not exist "%SCRATCH%\setup.py" ( echo ABORT: no setup.py in %SCRATCH% & exit /b 2 )
if not exist "%PYTHON%" ( echo ABORT: no venv python & exit /b 2 )

cd /d %SCRATCH%

set "PATH=%VSPATH%\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\%SDK%\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
set "INCLUDE=%VSPATH%\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\%SDK%\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\%SDK%\um;C:\Program Files (x86)\Windows Kits\10\Include\%SDK%\shared;C:\Program Files (x86)\Windows Kits\10\Include\%SDK%\winrt"
set "LIB=%VSPATH%\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%SDK%\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%SDK%\um\x64"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=8.9"
set MAX_JOBS=4
set "CUDAHOSTCXX=cl.exe"

echo ==== SPECTRE WHEEL BUILD START %DATE% %TIME% ==== > %L%
where cl >> %L% 2>&1
where nvcc >> %L% 2>&1
nvcc --version >> %L% 2>&1
"%PYTHON%" -c "import torch;print('TORCH',torch.__version__,torch.version.cuda)" >> %L% 2>&1
"%PYTHON%" -c "import exllamav3.version as v;print('BUILDING',v.__version__)" >> %L% 2>&1

"%PYTHON%" setup.py bdist_wheel --dist-dir dist >> %L% 2>&1
set RC=%ERRORLEVEL%
echo WHEEL-EXIT %RC% %DATE% %TIME% >> %L%
echo ==== SPECTRE WHEEL BUILD END %DATE% %TIME% ==== >> %L%
if exist dist\*.whl ( dir /b dist\*.whl )
exit /b %RC%
