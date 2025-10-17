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

---

# Hand Paddle Pong (Hand Hockey)

## Features
- Hand-tracking paddle control (MediaPipe, OpenCV)
- Neon retro style: glowing paddles, ball, and scoreboard
- Scoreboard with glowing numbers (now positioned lower for better visibility)
- Score glow effect when a player scores
- Paddle glow effect on hit
- Ball pause, wiggle, and twinkle after scoring
- Ball speed increases after each score
- Start screen with logo (press SPACE to start)
- Mute (M), volume (+/-), and fullscreen toggle

## Controls
- **SPACE**: Start game from title screen
- **ESC**: Quit
- **M**: Mute/unmute music
- **+ / -**: Volume up/down
- **P**: Play test tone
- **H/I/F**: Toggle hand mapping, swap hands, or mirror camera

## Visuals
- Scoreboard is now at Y=100 (moved down for better layout)
- Score glows yellow briefly when a player scores (matches paddle glow style)
- Paddles and ball have neon glow effects

# Multiball Experimental Version

## hand_pong_pygame_2.py
- Chaos multiball mode: every 15 seconds, spawns 3, then 4, then 5, ... balls at once
- Each extra ball gets a random neon color and random direction
- All multiballs have independent physics, scoring, and removal
- Main ball retains wiggle/twinkle effect; extra balls use neon style
- No cap on number of balls—game gets wilder over time!

### How to Run
To play the multiball version, run:

    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame_2.py"

All other controls and features are the same as the original game.

---
