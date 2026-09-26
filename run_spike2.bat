@echo off
rem Match-Bee spike2 run (detached): %1 = attn|probe
set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn;C:\Users\yoho\Downloads\exllamav3-kvarn\eval
set EXL3_KVARN_TRITON=1
set EXL3_KVARN_TRITON_PARITY=1
cd /d C:\Users\yoho\Downloads\exllamav3-kvarn
C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_spike2_online.py %1 > spike2_%1.log 2>&1
