import cv2

# Tries camera indices 0-4, shows a preview from each one that works.
# Note which index shows your USB camera's feed, then use that number
# for CAMERA_INDEX in main.py.

for index in range(5):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret:
        cap.release()
        continue

    print(f"Camera index {index} is available -- press any key to check the next one")
    cv2.imshow(f"Camera index {index}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

print("Done. Set CAMERA_INDEX in main.py to whichever index showed your USB camera.")

