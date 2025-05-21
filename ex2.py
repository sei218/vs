from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")


results = model("ex1.jpg")
nodes = results[0].keypoints.data[0][:, :2]

links = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12]
]

img = results[0].orig_img

for n1, n2 in links:
    if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
        continue
    cv2.line(
        img,
        nodes[n1].to(torch.int).tolist(),
        nodes[n2].to(torch.int).tolist(),
        (0, 0, 255),
        thickness=2,
    )

for idx, (x, y) in enumerate(nodes):
    if x == 0 or y == 0:
        continue
    if idx in [0, 1, 2, 3, 4]: 
        continue
    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)

points = [nodes[i] for i in [5, 6, 11, 12]]
valid_points = [p for p in points if p[0] != 0 and p[1] != 0]

if len(valid_points) == 4:
    avg_x = int(sum([p[0] for p in valid_points]) / 4)
    avg_y = int(sum([p[1] for p in valid_points]) / 4)
    cv2.circle(img, (avg_x, avg_y), 6, (255, 0, 0), -1) 

# 画像を表示
cv2.imshow("Pose with Center Point", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
