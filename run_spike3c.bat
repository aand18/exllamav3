@echo off
set PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn;C:\Users\yoho\Downloads\exllamav3-kvarn\eval
set EXL3_KVARN_TRITON=1
set EXL3_KVARN_TRITON_PARITY=1
cd /d C:\Users\yoho\Downloads\exllamav3-kvarn
C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe eval/_spike3_online.py probe2 > spike3_probe2.log 2>&1
