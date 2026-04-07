@echo off
cd /d C:\Users\taker\stock-dashboard
call .venv\Scripts\activate.bat
uv run python batch.py >> data\batch_log.txt 2>&1
uv run python analysis.py >> data\batch_log.txt 2>&1
echo %date% %time% バッチ完了 >> data\batch_log.txt
