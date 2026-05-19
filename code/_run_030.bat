@echo off
cd /d "C:\Users\qumin\projects\zhihu-econometrics"
python code\030_learning_roadmap.py
if exist images\030-roadmap.png (
    echo SUCCESS: Image generated
) else (
    echo FAIL: Image not found
)
