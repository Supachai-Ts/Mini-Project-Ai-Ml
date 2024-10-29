import cv2  # นำเข้า OpenCV สำหรับการจัดการวิดีโอและภาพ
import mediapipe as mp  # นำเข้า MediaPipe สำหรับการตรวจจับมือ
from hand_detection import HandDetector  # นำเข้า HandDetector จากโมดูล hand_detection

handDetector = HandDetector()  # สร้างอ็อบเจ็กต์ของ HandDetector

cap = cv2.VideoCapture(0)  # เปิดกล้องที่มีหมายเลข 0 (กล้องแรกที่เชื่อมต่อ)

fingers_index = {  # สร้างพจนานุกรมเพื่อเก็บชื่อของจำนวนมือที่ยกขึ้น
    0: "zero",  # 0 นิ้ว
    1: "one",   # 1 นิ้ว
    2: "two",   # 2 นิ้ว
    3: "three", # 3 นิ้ว
    4: "four",  # 4 นิ้ว
    5: "five"   # 5 นิ้ว
}

while True:  # ทำงานในลูปไม่รู้จบ
    success, img = cap.read()  # อ่านภาพจากกล้อง
    # ถ้าสำเร็จ จะส่งค่าภาพไปยังตัวแปร img

    img = handDetector.find_hands(img)  # ตรวจจับมือในภาพ

    land_mark_list = handDetector.find_position(img, draw=False)  # ค้นหาตำแหน่งของจุดบนมือ โดยไม่ต้องวาด

    fingers_up = handDetector.fingers_up()  # ตรวจสอบว่านิ้วไหนยกขึ้น

    img = cv2.flip(img, 1)  # พลิกภาพในแนวนอน

    if fingers_up is not None:  # ถ้ามีการตรวจจับนิ้ว
        max_fingers_up_count = fingers_up.count(1)  # นับจำนวนที่ยกนิ้วขึ้น

        # แสดงจำนวนที่ยกนิ้วบนภาพ
        cv2.putText(img, f'Counting: {str(fingers_index.get((max_fingers_up_count)))}', 
                    (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('image', img)  # แสดงภาพในหน้าต่างชื่อ 'image'
    
    if cv2.waitKey(1) & 0xff == ord('q'):  # ถ้ากด 'q' เพื่อออกจากลูป
        break  # ออกจากลูป
