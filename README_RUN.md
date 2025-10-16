Run instructions (prepared venv with mediapipe):

1) Use the included Python 3.11 virtual environment (already created at `.venv_py311`).

2) To start the game in windowed mode, double-click `run_game_v3.11.bat` or run in PowerShell:

    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame.py"

3) To run fullscreen:

    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame.py" --fullscreen

Notes:
- The venv has mediapipe, opencv-python, pygame, and numpy installed.
- If you want to use a different interpreter, install mediapipe into that interpreter first.
- If camera is off, the HUD will show CAMERA=OFF and the game will use a dummy black background.
