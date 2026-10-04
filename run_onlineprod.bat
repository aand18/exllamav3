@echo off
set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn;C:\Users\yoho\Downloads\exllamav3-kvarn\eval
set EXL3_KVARN_TRITON=1
cd /d C:\Users\yoho\Downloads\exllamav3-kvarn
C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_dbg_onlineprod.py > spike3_onlineprod.log 2>&1
