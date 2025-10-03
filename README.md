# Interactive Experience - Hand Tracking Pong Game

A real-time hand tracking pong game built with Python, OpenCV, and MediaPipe. Control the paddles with your hands using computer vision!

## Features

- **Hand Tracking**: Uses MediaPipe to detect and track hand movements
- **Real-time Gameplay**: Smooth pong gameplay with physics simulation
- **Dual Control**: Left and right hands control different paddles
- **Fast Ball Speed**: Enhanced speed for challenging gameplay
- **Cross-platform**: Works on Windows, macOS, and Linux

## Files

- `testing_1.py` - Main hand tracking version (requires Python 3.11 or 3.12)
- `testing_1_mouse_version.py` - Mouse-controlled version (works with any Python version)
- `run_hand_tracking.bat` - Windows batch file to easily run the hand tracking version
- `.gitignore` - Git ignore file to exclude virtual environments and large files

## Requirements

### For Hand Tracking Version
- Python 3.11 or 3.12 (MediaPipe compatibility)
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

### For Mouse Version
- Python 3.8+ (any recent version)
- OpenCV (`pip install opencv-python`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/famahung/Interactive_Experience.git
   cd Interactive_Experience
   ```

2. **Option A: Hand Tracking Version**
   ```bash
   # Create virtual environment with Python 3.11
   python3.11 -m venv venv_py311
   
   # Activate virtual environment
   # Windows:
   venv_py311\Scripts\activate
   # macOS/Linux:
   source venv_py311/bin/activate
   
   # Install dependencies
   pip install opencv-python mediapipe
   
   # Run the game
   python testing_1.py
   ```

3. **Option B: Mouse Version (Easier Setup)**
   ```bash
   pip install opencv-python
   python testing_1_mouse_version.py
   ```

## How to Play

### Hand Tracking Version
1. Make sure you have good lighting
2. Position yourself about 2-3 feet from your camera
3. Hold up both hands in front of the camera
4. Move your left hand to control the left paddle
5. Move your right hand to control the right paddle
6. The ball will bounce between paddles - try to keep it in play!
7. Press ESC to exit

### Mouse Version
1. Move your mouse to control the left paddle
2. The right paddle is controlled automatically (AI opponent)
3. Try to prevent the ball from going past your paddle
4. Press ESC to exit

## Game Mechanics

- Ball speed: Enhanced for faster gameplay
- Ball resets to center when it goes off-screen
- Paddle positions are constrained to stay within the screen
- Random ball direction after reset for unpredictability
- Collision detection with proper physics

## Troubleshooting

**MediaPipe won't install**: Make sure you're using Python 3.11 or 3.12. MediaPipe doesn't support Python 3.13 yet.

**Camera not working**: Check that your camera permissions are enabled and no other applications are using the camera.

**Hand detection issues**: Ensure good lighting and hold your hands clearly in front of the camera with palms facing the screen.

## Development

This project was created as part of a Creative Programming / Interactive Experience course. It demonstrates:
- Computer vision and hand tracking
- Real-time interactive applications  
- Game physics and collision detection
- Python multimedia programming

## License

This project is open source and available under the [MIT License](LICENSE).