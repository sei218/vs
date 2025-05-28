import cv2
import torch
from ultralytics import YOLO
model = YOLO("yolov8x.pt")
results = model.predict("ex2.jpg", conf=0.1)
img = results[0].orig_img.copy()
boxes = results[0].boxes

max_area = 0
max_box = None

# 最大面積探索
for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    width = xy2[0] - xy1[0]
    height = xy2[1] - xy1[1]
    area = width * height

    if area > max_area:
        max_area = area
        max_box = (xy1, xy2)

# 最大領域のみ描画
if max_box:
    xy1, xy2 = max_box
    cv2.rectangle(
        img,
        xy1.to(torch.int).tolist(),
        xy2.to(torch.int).tolist(),
        (0, 0, 255),
        thickness=3,
    )

cv2.imshow("Max Area Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
