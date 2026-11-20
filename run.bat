@echo off
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -r requirements.txt -q
python app.py
