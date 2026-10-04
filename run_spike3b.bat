@echo off
rem Match-Bee spike3b run (detached): %1 = attn|probe|qwht
set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn;C:\Users\yoho\Downloads\exllamav3-kvarn\eval
set EXL3_KVARN_TRITON=1
set EXL3_KVARN_TRITON_PARITY=1
cd /d C:\Users\yoho\Downloads\exllamav3-kvarn
C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_spike3_online.py %1 > spike3_%1.log 2>&1
