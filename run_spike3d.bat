@echo off
set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn;C:\Users\yoho\Downloads\exllamav3-kvarn\eval
set EXL3_KVARN_TRITON=1
set EXL3_KVARN_TRITON_PARITY=1
cd /d C:\Users\yoho\Downloads\exllamav3-kvarn
C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_dbg_imageless.py > spike3_imageless.log 2>&1
