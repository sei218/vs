import cv2
import torch
from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO("yolov8x.pt")

# 推論実行
results = model.predict("ex2.jpg", conf=0.1)
img = results[0].orig_img
boxes = results[0].boxes
names = model.names

for box in boxes:
    cls_id = int(box.cls[0].item())
    if names[cls_id] != "person":
        continue

    x1, y1, x2, y2 = box.xyxy[0].to(torch.int).tolist()
    person_img = img[y1:y2, x1:x2]

    # 青色
    hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
    lower_blue = (100, 100, 50)
    upper_blue = (140, 255, 255)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 青の割合
    blue_ratio = cv2.countNonZero(mask) / (person_img.shape[0] * person_img.shape[1])

    # 青色の服と判断された場合に赤枠を描画
    if blue_ratio > 0.01:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

# 表示
cv2.imshow("Blue Dressed Persons", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
