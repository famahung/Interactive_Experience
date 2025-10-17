"""
hand_pong_pygame.py
A pygame-based hand-controlled pong using MediaPipe for hand tracking and OpenCV for camera capture.
Features:
- Fullscreen game with scaled ball/paddles.
- Start screen (SPACE to start), ESC to quit.
- Use hand edge (landmarks 5 & 17) for paddle control; falls back to mouse if MediaPipe unavailable.
- Background music via pygame (looks for 'game_bgm.mp3' in the same folder, loops).
- M (mute), + / - volume controls.

Run with your venv python: & ".\.venv\Scripts\python.exe" "./hand_pong_pygame.py"
"""

import sys
import os
import time
import random
import json

import cv2
import numpy as np
import pygame
import traceback
import pathlib

# Early debug prints
print('STARTUP: python executable=', sys.executable, flush=True)
print('STARTUP: cwd=', os.getcwd(), flush=True)
print('STARTUP: pygame version=', pygame.version.ver, flush=True)

# Try import mediapipe, else we'll fallback to mouse control
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

# --- Config ---
CAM_W, CAM_H = 640, 480  # camera capture size
FPS = 60
BGM_FILE = 'game_bgm.mp3'
CONFIG_FILE = 'controls_config.json'

# Initialize pygame
pygame.init()
# Try init mixer but don't crash if it fails
try:
    pygame.mixer.init()
    mixer_ok = True
except Exception as e:
    print('DEBUG: pygame.mixer.init() failed:', e, flush=True)
    mixer_ok = False

info = pygame.display.Info()
# Default to windowed mode to avoid fullscreen focus issues; use --fullscreen to force fullscreen
WINDOWED = '--fullscreen' not in sys.argv
if WINDOWED:
    SCREEN_W = min(1280, info.current_w - 100)
    SCREEN_H = min(720, info.current_h - 100)
    flags = 0
else:
    SCREEN_W, SCREEN_H = info.current_w, info.current_h
    flags = pygame.FULLSCREEN

# Create window
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
pygame.display.set_caption('Hand Paddle Pong (Pygame)')
clock = pygame.time.Clock()
print(
    f'DEBUG: Created pygame window {SCREEN_W}x{SCREEN_H} (windowed={WINDOWED})', flush=True)

# Try to bring the pygame window to the foreground on Windows so users see it


def bring_window_to_front():
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            info = pygame.display.get_wm_info()
            hwnd = info.get('window') or info.get('hwnd')
            if hwnd:
                user32 = ctypes.windll.user32
                SW_SHOW = 5
                try:
                    # Use integers for hwnd calls
                    user32.ShowWindow(wintypes.HWND(hwnd), SW_SHOW)
                    user32.SetForegroundWindow(wintypes.HWND(hwnd))
                except Exception:
                    # fallback to passing raw int
                    try:
                        user32.ShowWindow(int(hwnd), SW_SHOW)
                        user32.SetForegroundWindow(int(hwnd))
                    except Exception as e:
                        print(
                            'DEBUG: SetForegroundWindow fallback failed:', e, flush=True)
                        return
                print('DEBUG: Called SetForegroundWindow on hwnd=', hwnd, flush=True)
    except Exception as e:
        print('DEBUG: bring_window_to_front failed:', e, flush=True)


bring_window_to_front()
# Scale factor based on 480 baseline
scale = SCREEN_H / 480.0

# Game sizes (scaled)
BALL_RADIUS = max(12, int(22 * scale))  # Bigger ball
BALL_SPEED = max(7, int(16 * scale))    # Faster ball
PADDLE_WIDTH = max(8, int(15 * scale))
PADDLE_HEIGHT = max(40, int(100 * scale))
MIN_PADDLE_H = max(30, int(60 * scale))
MAX_PADDLE_H = max(60, int(120 * scale))

# Ball pause/wiggle and paddle height variables (global, before start_screen)
wiggle_duration = 30
pause_duration = 60
ball_pause_frames = 0
ball_wiggle_frames = 0
ball_wiggle_opacity = 255
# Paddle heights (fixed)
left_paddle_h = PADDLE_HEIGHT
right_paddle_h = PADDLE_HEIGHT

# Colors - Retro Neon Style
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (50, 150, 255)
BLACK = (0, 0, 0)

# Neon colors matching the Hand Hockey logo
NEON_PINK = (255, 20, 147)
NEON_CYAN = (0, 255, 255)
NEON_YELLOW = (255, 255, 0)
NEON_BLUE = (100, 200, 255)
NEON_PURPLE = (200, 100, 255)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# Check camera availability early and keep a flag
CAMERA_AVAILABLE = cap.isOpened()
if not CAMERA_AVAILABLE:
    print('DEBUG: Camera not opened (cap.isOpened() is False). Falling back to mouse-only control and dummy frames.', flush=True)

# MediaPipe setup
if MP_AVAILABLE:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)
else:
    hands = None

# BGM
bgm_muted = False
bgm_volume = 0.6
BGM_LOADED = False
if os.path.exists(BGM_FILE):
    try:
        pygame.mixer.music.load(BGM_FILE)
        pygame.mixer.music.set_volume(bgm_volume)
        BGM_LOADED = True
        print(
            f'DEBUG: Loaded BGM "{BGM_FILE}" and set volume to {bgm_volume}', flush=True)
    except Exception as e:
        print('Could not load BGM:', e)
        traceback.print_exc()
else:
    # no file; do nothing (avoid auto-generating here)
    pass

# Helper functions


def draw_neon_paddle(surface, rect, primary_color, glow_color):
    """Draw a paddle with neon glow effect"""
    # Draw outer glow layers
    for i in range(3, 0, -1):
        glow_rect = rect.inflate(i * 4, i * 4)
        glow_surf = pygame.Surface(
            (glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        alpha = 60 - (i * 15)
        pygame.draw.rect(glow_surf, (*glow_color, alpha),
                         glow_surf.get_rect(), border_radius=5)
        surface.blit(glow_surf, glow_rect.topleft)

    # Draw main paddle with gradient
    pygame.draw.rect(surface, primary_color, rect, border_radius=5)
    # Add bright edge/outline
    pygame.draw.rect(surface, glow_color, rect, 3, border_radius=5)


def draw_neon_ball(surface, x, y, radius, primary_color, glow_color):
    """Draw a ball with neon glow effect"""
    # Draw outer glow layers
    for i in range(4, 0, -1):
        glow_radius = radius + (i * 3)
        glow_surf = pygame.Surface(
            (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        alpha = 80 - (i * 15)
        pygame.draw.circle(glow_surf, (*glow_color, alpha),
                           (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(x - glow_radius), int(y - glow_radius)))

    # Draw main ball
    pygame.draw.circle(surface, primary_color, (int(x), int(y)), radius)
    # Add bright center
    pygame.draw.circle(surface, (255, 255, 255),
                       (int(x), int(y)), max(3, radius // 3))


def draw_text(surface, text, x, y, size, color=WHITE, center=False, bold=False):
    # Try gaming fonts first, fallback to Arial
    font_names = ['Consolas', 'Courier New', 'monospace', 'Arial']
    font = None
    for font_name in font_names:
        try:
            font = pygame.font.SysFont(font_name, size, bold=bold)
            break
        except:
            continue
    if font is None:
        font = pygame.font.Font(None, size)

    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(surf, rect)


def start_screen():
    # Try to load start screen image
    start_image = None
    start_image_path = 'hand_hockey_start.jpg'
    if os.path.exists(start_image_path):
        try:
            start_image = pygame.image.load(start_image_path)
            # Scale to fit screen while maintaining aspect ratio
            img_rect = start_image.get_rect()
            scale_factor = min(SCREEN_W / img_rect.width,
                               SCREEN_H / img_rect.height)
            new_width = int(img_rect.width * scale_factor)
            new_height = int(img_rect.height * scale_factor)
            start_image = pygame.transform.scale(
                start_image, (new_width, new_height))
            print(f'DEBUG: Loaded start screen image: {start_image_path}')
        except Exception as e:
            print(f'DEBUG: Failed to load start screen image: {e}')
            start_image = None

    # Start BGM when start screen appears
    if mixer_ok and pygame.mixer.get_init() and pygame.mixer.music.get_busy() == 0 and os.path.exists(BGM_FILE):
        try:
            pygame.mixer.music.play(-1)
            print('DEBUG: Started BGM playback on start screen (loop)', flush=True)
        except Exception as e:
            print(f'DEBUG: Failed to start BGM: {e}', flush=True)

    # Display start screen and wait for SPACE
    while True:
        if start_image:
            # Fill with dark background
            screen.fill((10, 10, 30))
            # Center the image
            img_rect = start_image.get_rect(
                center=(SCREEN_W//2, SCREEN_H//2 - 50))
            screen.blit(start_image, img_rect)
        else:
            # Fallback: Display live camera in background with overlay
            if CAMERA_AVAILABLE:
                ret, frame = cap.read()
                if not ret:
                    print('DEBUG: frame read failed in start_screen(); continuing loop')
                    frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            else:
                ret = True
                frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            if mirror_camera:
                frame = cv2.flip(frame, 1)
            # Convert to RGB and scale to screen size
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (SCREEN_W, SCREEN_H))
            pygame_frame = pygame.surfarray.make_surface(
                frame_rgb.swapaxes(0, 1))
            screen.blit(pygame_frame, (0, 0))

            # translucent overlay
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (0, 0, 0, 150),
                             (20, SCREEN_H//4, SCREEN_W-40, int(SCREEN_H*0.15)))
            screen.blit(overlay, (0, 0))

            draw_text(screen, 'Hand Paddle Pong Game', SCREEN_W//2,
                      SCREEN_H//3, int(48*scale), WHITE, center=True)

        # Instructions (shown for both image and camera background)
        draw_text(screen, 'Press SPACE to start — ESC to quit', SCREEN_W //
                  2, SCREEN_H - 150, int(28*scale), (255, 255, 255), center=True)
        draw_text(screen, 'M to mute, +/- volume', SCREEN_W//2,
                  SCREEN_H - 100, int(20*scale), (200, 200, 200), center=True)

        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_SPACE:
                    return True
        clock.tick(30)


# Control mapping options with persisted config - LOAD EARLY
# Defaults
use_handedness_mapping = False  # Use screen position, not MediaPipe's hand labels
swap_hands = False
mirror_camera = True  # Mirror ON for selfie view (like looking in a mirror)
# Try load from config
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            if isinstance(cfg, dict):
                use_handedness_mapping = bool(
                    cfg.get('use_handedness_mapping', use_handedness_mapping))
                swap_hands = bool(cfg.get('swap_hands', swap_hands))
                mirror_camera = bool(cfg.get('mirror_camera', mirror_camera))
except Exception as e:
    print('DEBUG: Failed to load config:', e, flush=True)


def save_config():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'use_handedness_mapping': use_handedness_mapping,
                'swap_hands': swap_hands,
                'mirror_camera': mirror_camera
            }, f)
    except Exception as e:
        print('DEBUG: Failed to save config:', e, flush=True)

        # Update score
        if ball_x < 0:
            right_score += 1
            print(
                f'DEBUG: Right player scores! Score: {left_score} - {right_score}', flush=True)
        else:
            left_score += 1
            print(
                f'DEBUG: Left player scores! Score: {left_score} - {right_score}', flush=True)

        # Gradually increase ball speed
        ball_speed_multiplier = min(ball_speed_multiplier + 0.12, 3.0)

        # Reset ball position and pause
        ball_x, ball_y = SCREEN_W//2, SCREEN_H//2
        ball_dx = BALL_SPEED if random.choice([True, False]) else -BALL_SPEED
        ball_dy = random.choice([
            -BALL_SPEED, -int(BALL_SPEED*0.8), int(BALL_SPEED*0.8), BALL_SPEED])
        ball_pause_frames = pause_duration
        ball_wiggle_frames = wiggle_duration
        ball_wiggle_opacity = 255


# Paddle glow effect state
left_paddle_glow = 0
right_paddle_glow = 0
PADDLE_GLOW_FRAMES = 18

use_mouse_fallback = not MP_AVAILABLE

if not start_screen():
    cap.release()
    pygame.quit()
    sys.exit(0)
print('DEBUG: start_screen returned True, entering main loop', flush=True)

# Initialize game state variables
ball_x, ball_y = SCREEN_W//2, SCREEN_H//2
ball_dx = BALL_SPEED if random.choice([True, False]) else -BALL_SPEED
ball_dy = random.choice(
    [-BALL_SPEED, -int(BALL_SPEED*0.8), int(BALL_SPEED*0.8), BALL_SPEED])
ball_speed_multiplier = 1.0
left_paddle_y = SCREEN_H // 2
right_paddle_y = SCREEN_H // 2
left_score = 0
right_score = 0
wiggle_offset_x = 0
wiggle_offset_y = 0
# Score glow effect
left_score_glow = 0
right_score_glow = 0
SCORE_GLOW_FRAMES = 45

# Main loop
running = True
last_heartbeat = time.time()
# make sure cursor is visible so user can see it's interactive
pygame.mouse.set_visible(True)
while running:
    # Read camera frame (or use dummy if camera not available)
    if CAMERA_AVAILABLE:
        ret, frame = cap.read()
        if not ret:
            print('DEBUG: frame read failed in main loop; using black frame', flush=True)
            frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    else:
        frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    if mirror_camera:
        frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # default paddles remain unchanged unless hand detected
    if MP_AVAILABLE and hands:
        # MediaPipe recommends making the image non-writeable to improve performance
        try:
            img_rgb.flags.writeable = False
            results = hands.process(img_rgb)
            img_rgb.flags.writeable = True
        except Exception as e:
            results = None
            print('DEBUG: hands.process failed:', e, flush=True)

        if results and results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # use landmarks 5 and 17 as the side-edge points
                edge_top = hand_landmarks.landmark[5]
                edge_bottom = hand_landmarks.landmark[17]
                # normalized midpoint
                norm_x = (edge_top.x + edge_bottom.x) / 2.0
                norm_y = (edge_top.y + edge_bottom.y) / 2.0
                # map normalized coordinates directly to screen (camera now fills full screen)
                edge_cx = int(norm_x * SCREEN_W)
                edge_cy = int(norm_y * SCREEN_H)
                edge_len = int(abs(edge_top.y - edge_bottom.y) * SCREEN_H)
                dyn_h = max(MIN_PADDLE_H, min(MAX_PADDLE_H, edge_len * 2))

                # Determine intended paddle target based on mode
                target = None  # 'left' or 'right'
                hand_label = None
                try:
                    if handedness and hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                        # 'Left' or 'Right'
                        hand_label = handedness.classification[0].label
                except Exception:
                    hand_label = None

                if use_handedness_mapping and hand_label in ("Left", "Right"):
                    target = 'left' if hand_label == 'Left' else 'right'
                else:
                    # Fallback: choose by on-screen side of the midpoint (intuitive mapping)
                    # Left half of screen -> left paddle, Right half -> right paddle
                    target = 'left' if edge_cx < SCREEN_W // 2 else 'right'

                if swap_hands:
                    target = 'left' if target == 'right' else 'right'

                if target == 'left':
                    left_paddle_y = max(
                        left_paddle_h//2, min(SCREEN_H - left_paddle_h//2, edge_cy))
                    print(
                        f"DEBUG: LEFT paddle - hand at ({edge_cx},{edge_cy}) label={hand_label} mode={'handedness' if use_handedness_mapping else 'screen'} swap={swap_hands}", flush=True)
                else:
                    right_paddle_y = max(
                        right_paddle_h//2, min(SCREEN_H - right_paddle_h//2, edge_cy))
                    print(
                        f"DEBUG: RIGHT paddle - hand at ({edge_cx},{edge_cy}) label={hand_label} mode={'handedness' if use_handedness_mapping else 'screen'} swap={swap_hands}", flush=True)
        else:
            # no hands detected this frame
            pass
    else:
        # mouse fallback (control right paddle)
        mx, my = pygame.mouse.get_pos()
        right_paddle_y = my

    # Ball pause and wiggle effect after score
    if ball_pause_frames > 0:
        ball_pause_frames -= 1
        # Wiggle effect: shake ball and change opacity
        wiggle_phase = (pause_duration - ball_pause_frames)
        if wiggle_phase < wiggle_duration:
            ball_wiggle_frames = wiggle_duration - wiggle_phase
            ball_wiggle_opacity = int(
                120 + 80 * abs((wiggle_phase % 8) - 4) / 4)
            # Ball wiggle offset
            wiggle_offset_x = random.randint(-6, 6)
            wiggle_offset_y = random.randint(-6, 6)
        else:
            ball_wiggle_frames = 0
            ball_wiggle_opacity = 255
            wiggle_offset_x = 0
            wiggle_offset_y = 0
    else:
        # Move ball normally
        ball_x += ball_dx * ball_speed_multiplier
        ball_y += ball_dy * ball_speed_multiplier
        ball_wiggle_frames = 0
        ball_wiggle_opacity = 255
        wiggle_offset_x = 0
        wiggle_offset_y = 0

    # Collisions top/bottom
    if ball_y - BALL_RADIUS < 0 or ball_y + BALL_RADIUS > SCREEN_H:
        ball_dy *= -1

    # Left paddle collision
    if ball_x - BALL_RADIUS < PADDLE_WIDTH and abs(ball_y - left_paddle_y) < left_paddle_h//2:
        ball_dx *= -1
        # Add rebound force based on paddle movement (left paddle)
        ball_dy += (left_paddle_y - ball_y) * 0.08
        left_paddle_glow = PADDLE_GLOW_FRAMES
    # Right paddle collision
    if ball_x + BALL_RADIUS > SCREEN_W - PADDLE_WIDTH and abs(ball_y - right_paddle_y) < right_paddle_h//2:
        ball_dx *= -1
        # Add rebound force based on paddle movement (right paddle)
        ball_dy += (right_paddle_y - ball_y) * 0.08
        right_paddle_glow = PADDLE_GLOW_FRAMES

    # Out of bounds reset and score tracking
    if ball_x < 0 or ball_x > SCREEN_W:
        # Update score
        if ball_x < 0:
            right_score += 1
            right_score_glow = SCORE_GLOW_FRAMES
            print(
                f'DEBUG: Right player scores! Score: {left_score} - {right_score}', flush=True)
        else:
            left_score += 1
            left_score_glow = SCORE_GLOW_FRAMES
            print(
                f'DEBUG: Left player scores! Score: {left_score} - {right_score}', flush=True)

        # Gradually increase ball speed
        ball_speed_multiplier = min(ball_speed_multiplier + 0.12, 3.0)

        # Reset ball position and pause
        ball_x, ball_y = SCREEN_W//2, SCREEN_H//2
        ball_dx = BALL_SPEED if random.choice([True, False]) else -BALL_SPEED
        ball_dy = random.choice([
            -BALL_SPEED, -int(BALL_SPEED*0.8), int(BALL_SPEED*0.8), BALL_SPEED])
        ball_pause_frames = pause_duration
        ball_wiggle_frames = wiggle_duration
        ball_wiggle_opacity = 255

    # Draw
    # Draw camera as background (maintain aspect ratio, fill screen)
    if CAMERA_AVAILABLE:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_h, cam_w = frame_rgb.shape[:2]
        # Scale to cover the entire screen (may crop edges to maintain aspect ratio)
        scale_w = SCREEN_W / cam_w
        scale_h = SCREEN_H / cam_h
        # use max to fill screen (crop if needed)
        scale = max(scale_w, scale_h)
        new_w = int(cam_w * scale)
        new_h = int(cam_h * scale)
        resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Center the camera (crop edges if larger than screen)
        offset_x = (new_w - SCREEN_W) // 2
        offset_y = (new_h - SCREEN_H) // 2
        cropped = resized[offset_y:offset_y +
                          SCREEN_H, offset_x:offset_x+SCREEN_W]

        # pygame needs height x width x channels, so swap axes like start screen
        bg_surf = pygame.surfarray.make_surface(cropped.swapaxes(0, 1))
        screen.blit(bg_surf, (0, 0))
    else:
        # Dark retro background
        screen.fill((10, 10, 30))

    # Draw neon ball with glow effect
    # Draw neon ball with wiggle and opacity effect
    ball_draw_x = int(ball_x + wiggle_offset_x)
    ball_draw_y = int(ball_y + wiggle_offset_y)
    # Draw glow layers with changing opacity if paused
    if ball_pause_frames > 0 and ball_wiggle_frames > 0:
        # Twinkle effect: change color rapidly
        twinkle_colors = [NEON_YELLOW, NEON_CYAN,
                          NEON_PINK, NEON_PURPLE, WHITE]
        twinkle_idx = ((pause_duration - ball_pause_frames) //
                       3) % len(twinkle_colors)
        twinkle_color = twinkle_colors[twinkle_idx]
        for i in range(4, 0, -1):
            glow_radius = BALL_RADIUS + (i * 3)
            glow_surf = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = max(40, int(ball_wiggle_opacity * (0.7 - i * 0.1)))
            pygame.draw.circle(glow_surf, (*twinkle_color, alpha),
                               (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surf, (ball_draw_x - glow_radius,
                        ball_draw_y - glow_radius))
        pygame.draw.circle(screen, twinkle_color,
                           (ball_draw_x, ball_draw_y), BALL_RADIUS)
        # Twinkle center highlight
        pygame.draw.circle(screen, WHITE, (ball_draw_x,
                           ball_draw_y), max(3, BALL_RADIUS // 3))
    else:
        draw_neon_ball(screen, ball_draw_x, ball_draw_y,
                       BALL_RADIUS, NEON_YELLOW, NEON_CYAN)

    # Draw neon paddles with glow effect
    # Left paddle - Pink/Cyan gradient
    left_rect = pygame.Rect(
        0, int(left_paddle_y - left_paddle_h//2), PADDLE_WIDTH, int(left_paddle_h))
    glow_strength_left = left_paddle_glow / \
        PADDLE_GLOW_FRAMES if left_paddle_glow > 0 else 0
    if left_paddle_glow > 0:
        for i in range(4, 0, -1):
            glow_rect = left_rect.inflate(i * 8, i * 8)
            glow_surf = pygame.Surface(
                (glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            alpha = int(80 * glow_strength_left * (1 - i * 0.18))
            pygame.draw.rect(glow_surf, (*NEON_YELLOW, alpha),
                             glow_surf.get_rect(), border_radius=7)
            screen.blit(glow_surf, glow_rect.topleft)
    draw_neon_paddle(screen, left_rect, NEON_PINK, NEON_CYAN)
    if left_paddle_glow > 0:
        left_paddle_glow -= 1

    # Right paddle - Cyan/Pink gradient (opposite)
    right_rect = pygame.Rect(SCREEN_W - PADDLE_WIDTH, int(right_paddle_y -
                             right_paddle_h//2), PADDLE_WIDTH, int(right_paddle_h))
    glow_strength_right = right_paddle_glow / \
        PADDLE_GLOW_FRAMES if right_paddle_glow > 0 else 0
    if right_paddle_glow > 0:
        for i in range(4, 0, -1):
            glow_rect = right_rect.inflate(i * 8, i * 8)
            glow_surf = pygame.Surface(
                (glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            alpha = int(80 * glow_strength_right * (1 - i * 0.18))
            pygame.draw.rect(glow_surf, (*NEON_YELLOW, alpha),
                             glow_surf.get_rect(), border_radius=7)
            screen.blit(glow_surf, glow_rect.topleft)
    draw_neon_paddle(screen, right_rect, NEON_CYAN, NEON_PINK)
    if right_paddle_glow > 0:
        right_paddle_glow -= 1

    # Draw hand landmarks (visual feedback for detected hands)
    if MP_AVAILABLE and hands and results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks 5 and 17
            lm5 = hand_landmarks.landmark[5]
            lm17 = hand_landmarks.landmark[17]
            # Map directly to full screen coords (camera now fills screen)
            x5 = int(lm5.x * SCREEN_W)
            y5 = int(lm5.y * SCREEN_H)
            x17 = int(lm17.x * SCREEN_W)
            y17 = int(lm17.y * SCREEN_H)

            # Draw dots and line with neon glow
            # Glow for landmark 5
            for i in range(3, 0, -1):
                alpha = 60 - (i * 15)
                glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surf, (*NEON_PINK, alpha), (15, 15), 12 + i * 2)
                screen.blit(glow_surf, (x5 - 15, y5 - 15))
            pygame.draw.circle(screen, NEON_PINK, (x5, y5), 12)
            pygame.draw.circle(screen, WHITE, (x5, y5), 5)

            # Glow for landmark 17
            for i in range(3, 0, -1):
                alpha = 60 - (i * 15)
                glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surf, (*NEON_CYAN, alpha), (15, 15), 12 + i * 2)
                screen.blit(glow_surf, (x17 - 15, y17 - 15))
            pygame.draw.circle(screen, NEON_CYAN, (x17, y17), 12)
            pygame.draw.circle(screen, WHITE, (x17, y17), 5)

            # Neon line connecting them
            pygame.draw.line(screen, NEON_YELLOW, (x5, y5), (x17, y17), 7)
            pygame.draw.line(screen, WHITE, (x5, y5), (x17, y17), 3)

    # Draw scoreboard at the top center with neon style
    score_font_size = int(72 * scale)

    # Left score (pink glow) - Player 1
    left_score_text = str(left_score)
    # Calculate glow strength
    left_glow_strength = left_score_glow / \
        SCORE_GLOW_FRAMES if left_score_glow > 0 else 0

    # Create glow layers with darker shades
    glow_layers = [
        (4, (100, 10, 60)),    # Outer glow
        (3, (150, 15, 90)),    # Mid glow
        (2, (200, 20, 120)),   # Inner glow
    ]

    # Extra glow when score changes (similar to paddle glow)
    if left_score_glow > 0:
        for i in range(4, 0, -1):
            alpha = int(80 * left_glow_strength * (1 - i * 0.18))
            draw_text(screen, left_score_text, SCREEN_W//2 - 100, 100,
                      score_font_size + i * 4, (*NEON_YELLOW, alpha), center=True, bold=True)
        left_score_glow -= 1

    for offset, glow_color in glow_layers:
        draw_text(screen, left_score_text, SCREEN_W//2 - 100, 100,
                  score_font_size + offset * 2, glow_color, center=True, bold=True)
    # Main score
    draw_text(screen, left_score_text, SCREEN_W//2 - 100, 100,
              score_font_size, NEON_PINK, center=True, bold=True)
    # Bright highlight
    draw_text(screen, left_score_text, SCREEN_W//2 - 100, 100,
              score_font_size - 10, WHITE, center=True, bold=True)

    # Right score (cyan glow) - Player 2
    right_score_text = str(right_score)
    # Calculate glow strength
    right_glow_strength = right_score_glow / \
        SCORE_GLOW_FRAMES if right_score_glow > 0 else 0

    # Create glow layers with darker shades
    glow_layers_cyan = [
        (4, (0, 100, 100)),    # Outer glow
        (3, (0, 150, 150)),    # Mid glow
        (2, (0, 200, 200)),    # Inner glow
    ]

    # Extra glow when score changes (similar to paddle glow)
    if right_score_glow > 0:
        for i in range(4, 0, -1):
            alpha = int(80 * right_glow_strength * (1 - i * 0.18))
            draw_text(screen, right_score_text, SCREEN_W//2 + 100, 100,
                      score_font_size + i * 4, (*NEON_YELLOW, alpha), center=True, bold=True)
        right_score_glow -= 1

    for offset, glow_color in glow_layers_cyan:
        draw_text(screen, right_score_text, SCREEN_W//2 + 100, 100,
                  score_font_size + offset * 2, glow_color, center=True, bold=True)
    # Main score
    draw_text(screen, right_score_text, SCREEN_W//2 + 100, 100,
              score_font_size, NEON_CYAN, center=True, bold=True)
    # Bright highlight
    draw_text(screen, right_score_text, SCREEN_W//2 + 100, 100,
              score_font_size - 10, WHITE, center=True, bold=True)

    pygame.display.flip()

    # Events
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_p:
                # play a short test tone to verify audio
                if mixer_ok:
                    try:
                        freq = 440
                        duration_s = 0.3
                        sample_rate = 44100
                        t = np.linspace(0, duration_s, int(
                            sample_rate * duration_s), False)
                        tone = np.sin(freq * t * 2 * np.pi)
                        audio = np.int16(tone * 32767)
                        snd = pygame.sndarray.make_sound(audio)
                        snd.play()
                        print('DEBUG: Played test tone (P)', flush=True)
                    except Exception as e:
                        print('DEBUG: Failed to play test tone:', e, flush=True)
                else:
                    print(
                        'DEBUG: Mixer not available, cannot play test tone', flush=True)
            elif ev.key == pygame.K_m:
                bgm_muted = not bgm_muted
                if bgm_muted:
                    pygame.mixer.music.set_volume(0.0)
                else:
                    pygame.mixer.music.set_volume(bgm_volume)
            elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                bgm_volume = min(1.0, bgm_volume + 0.1)
                if not bgm_muted:
                    pygame.mixer.music.set_volume(bgm_volume)
            elif ev.key == pygame.K_MINUS:
                bgm_volume = max(0.0, bgm_volume - 0.1)
                if not bgm_muted:
                    pygame.mixer.music.set_volume(bgm_volume)
            elif ev.key == pygame.K_h:
                # Toggle mapping mode between handedness and screen-side
                use_handedness_mapping = not use_handedness_mapping
            elif ev.key == pygame.K_i:
                # Invert mapping assignments
                swap_hands = not swap_hands
            elif ev.key == pygame.K_f:
                # Toggle camera mirroring
                mirror_camera = not mirror_camera
            # Persist mapping preferences whenever changed
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_h, pygame.K_i, pygame.K_f):
                try:
                    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                        json.dump({
                            'use_handedness_mapping': use_handedness_mapping,
                            'swap_hands': swap_hands,
                            'mirror_camera': mirror_camera
                        }, f)
                except Exception as e:
                    print('DEBUG: Failed to save config:', e, flush=True)

    clock.tick(FPS)

    # periodic heartbeat so we know loop is alive
    if time.time() - last_heartbeat > 3.0:
        print('DEBUG: main loop heartbeat', flush=True)
        last_heartbeat = time.time()

# Cleanup
cap.release()
stop = False
try:
    pygame.mixer.music.stop()
except Exception:
    pass
pygame.quit()
sys.exit(0)
