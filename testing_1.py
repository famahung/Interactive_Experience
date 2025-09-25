import cv2
import mediapipe as mp
import random

# Mediapipe 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 視窗大小
width, height = 640, 480

# 球初始參數
ball_x, ball_y = width // 2, height // 2
ball_dx, ball_dy = 5, 5
ball_radius = 15

# Paddle 參數
paddle_height = 100
paddle_width = 15
left_paddle_y, right_paddle_y = height // 2, height // 2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 鏡像，方便控制
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 取得手掌座標（landmark 0 = 腕部）
            h, w, _ = frame.shape
            cx, cy = int(
                hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)

            if handedness.classification[0].label == "Left":
                left_paddle_y = max(
                    paddle_height//2, min(height - paddle_height//2, cy))
            else:
                right_paddle_y = max(
                    paddle_height//2, min(height - paddle_height//2, cy))

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    # 移動球
    ball_x += ball_dx
    ball_y += ball_dy

    # 撞到上下邊
    if ball_y - ball_radius < 0 or ball_y + ball_radius > height:
        ball_dy *= -1

    # 左右板碰撞
    if (ball_x - ball_radius < paddle_width and abs(ball_y - left_paddle_y) < paddle_height // 2) or \
       (ball_x + ball_radius > width - paddle_width and abs(ball_y - right_paddle_y) < paddle_height // 2):
        ball_dx *= -1

    # 球超出左右邊界時重置（遊戲重新開始）
    if ball_x < 0 or ball_x > width:
        ball_x, ball_y = width // 2, height // 2
        ball_dx = 5 if random.choice([True, False]) else -5
        ball_dy = random.choice([-5, -3, 3, 5])

    # 畫球
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 255), -1)

    # 畫左板
    cv2.rectangle(frame, (0, left_paddle_y - paddle_height//2),
                  (paddle_width, left_paddle_y + paddle_height//2), (0, 255, 0), -1)
    # 畫右板
    cv2.rectangle(frame, (width - paddle_width, right_paddle_y - paddle_height//2),
                  (width, right_paddle_y + paddle_height//2), (0, 0, 255), -1)

    cv2.imshow("手控乒乓球", cv2.resize(frame, (width, height)))

    if cv2.waitKey(1) & 0xFF == 27:  # ESC鍵退出
        break

cap.release()
cv2.destroyAllWindows()
