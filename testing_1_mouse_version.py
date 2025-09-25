import cv2
import random

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

# 滑鼠控制變數
mouse_x, mouse_y = width // 2, height // 2


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


cap = cv2.VideoCapture(0)
cv2.namedWindow("手控乒乓球 (滑鼠版)")
cv2.setMouseCallback("手控乒乓球 (滑鼠版)", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 鏡像，方便控制

    # 使用滑鼠控制板子（左邊板子跟隨滑鼠Y座標）
    left_paddle_y = max(
        paddle_height//2, min(height - paddle_height//2, mouse_y))
    # 右邊板子可以用鍵盤控制或設為自動
    # 這裡設為簡單的自動跟隨球的Y座標
    right_paddle_y = max(
        paddle_height//2, min(height - paddle_height//2, ball_y))

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

    # 畫左板（玩家控制）
    cv2.rectangle(frame, (0, left_paddle_y - paddle_height//2),
                  (paddle_width, left_paddle_y + paddle_height//2), (0, 255, 0), -1)
    # 畫右板（電腦控制）
    cv2.rectangle(frame, (width - paddle_width, right_paddle_y - paddle_height//2),
                  (width, right_paddle_y + paddle_height//2), (0, 0, 255), -1)

    # 顯示提示文字
    cv2.putText(frame, "Move mouse to control left paddle", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press ESC to exit", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("手控乒乓球 (滑鼠版)", cv2.resize(frame, (width, height)))

    if cv2.waitKey(1) & 0xFF == 27:  # ESC鍵退出
        break

cap.release()
cv2.destroyAllWindows()
