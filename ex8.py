from ultralytics import YOLO

model=YOLO("best.pt")

results=model.predict("ex4.jpg", conf=0.7)
labels=results[0].boxes.cls.tolist()
#カウント
count_white = labels.count(0)
count_black = labels.count(1)

print(f"白の石:{count_white}個")
print(f"黒の石:{count_black}個")
