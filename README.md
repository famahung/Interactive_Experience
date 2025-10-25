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

# Game Versions

## hand_pong_pygame.py (Original Version)
The classic hand-controlled Pong game:
- Traditional gameplay: paddles locked to left and right sides of the screen
- Paddles move only vertically (up/down)
- Single ball gameplay
- Hand tracking controls paddle vertical position
- Classic Pong-style collision detection

### How to Run
    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame.py"

---

## hand_pong_pygame_2.py (Multiball Chaos Version)
Experimental multiball mode with traditional paddle mechanics:
- **Multiball chaos**: Every 15 seconds, spawns 3, then 4, then 5, ... balls at once
- Each extra ball gets a random neon color and random direction
- All multiballs have independent physics, scoring, and removal
- Main ball retains wiggle/twinkle effect; extra balls use neon style
- No cap on number of balls—game gets wilder over time!
- Paddles still locked to left/right sides (classic Pong style)

### How to Run
    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame_2.py"

---

## hand_pong_pygame_3.py (Free-Moving Paddle Version)
Revolutionary gameplay with full 2D paddle control:
- **Free-moving paddles**: Paddles can move anywhere on the screen (X and Y axes)
- Hand tracking maps to full screen coordinates - move your hand anywhere!
- Circular collision detection: Ball bounces away from paddle center based on angle of impact
- More dynamic, air-hockey-style gameplay
- Multiball mode included (same chaos as version 2)
- Strategic positioning becomes crucial - block and defend from any angle!

### How to Run
    & ".\.venv_py311\Scripts\python.exe" ".\hand_pong_pygame_3.py"

### Key Differences from Version 2:
- **Paddle Movement**: Full 2D freedom vs locked to sides
- **Collision Physics**: Circular/angular bounce vs edge-based reflection
- **Gameplay Style**: Air hockey meets Pong vs traditional Pong
- **Strategy**: Positional play and interception vs timing and reflexes

---
