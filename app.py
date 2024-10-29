# นำเข้า Flask สำหรับสร้างแอปเว็บ, render_template สำหรับการเรนเดอร์ HTML, และ Response สำหรับส่งข้อมูลไปยังไคลเอนต์
from flask import Flask, render_template, Response  
import cv2  # นำเข้า OpenCV สำหรับการจัดการวิดีโอและภาพ
from hand_detection import HandDetector  # นำเข้า HandDetector จากโมดูล hand_detection เพื่อใช้ในการตรวจจับมือ

app = Flask(__name__)  # สร้างแอปพลิเคชัน Flask
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

if not cap.isOpened():  # ตรวจสอบว่ากล้องเปิดได้หรือไม่
    print("ไม่สามารถเปิดกล้องได้")  # ถ้าไม่สามารถเปิดกล้องได้ จะแสดงข้อความนี้

def generate_frames():  # ฟังก์ชันที่สร้างเฟรมวิดีโอ
    while True:  # ทำงานในลูปไม่รู้จบ
        success, img = cap.read()  # อ่านภาพจากกล้อง
        if not success:  # ถ้าไม่สามารถอ่านภาพได้
            print("ไม่สามารถอ่านภาพจากกล้องได้")  # แสดงข้อความข้อผิดพลาด
            break  # ออกจากลูป

        img = handDetector.find_hands(img)  # ตรวจจับมือในภาพ
        land_mark_list = handDetector.find_position(img, draw=False)  # หาตำแหน่งจุดบนมือ โดยไม่ต้องวาด (draw=False)
        fingers_up = handDetector.fingers_up()  # ตรวจสอบว่านิ้วไหนยกขึ้น

        if fingers_up is not None:  # ถ้ามีการตรวจจับนิ้ว
            max_fingers_up_count = fingers_up.count(1)  # นับจำนวนที่นิ้วยกขึ้น
        else:  # ถ้าไม่พบการยกนิ้ว
            max_fingers_up_count = 0  # กำหนดเป็น 0

        img = cv2.flip(img, 1)  # พลิกภาพในแนวนอน

        # แสดงจำนวนที่ยกนิ้วบนภาพ
        cv2.putText(img, f'Counting: {str(fingers_index.get((max_fingers_up_count)))}', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', img)  # แปลงภาพเป็นรูปแบบ JPEG
        img_stream = buffer.tobytes()  # แปลงข้อมูลภาพเป็น byte stream

        yield (b'--frame\r\n'  # เริ่มต้นส่งเฟรม
               b'Content-Type: image/jpeg\r\n\r\n' + img_stream + b'\r\n')  # ส่งข้อมูลภาพในรูปแบบ JPEG

@app.route('/')  # กำหนดเส้นทางสำหรับหน้าแรก
def index():
    return render_template('index.html')  # เรนเดอร์เทมเพลต HTML ที่ชื่อ index.html

@app.route('/video_feed')  # กำหนดเส้นทางสำหรับฟีดวิดีโอ
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # ส่งข้อมูลเฟรมวิดีโอในรูปแบบที่สามารถเล่นได้

if __name__ == '__main__':  # ถ้าไฟล์นี้ถูกเรียกใช้งานโดยตรง
    app.run(debug=True)  # รันแอปพลิเคชัน Flask โดยเปิดโหมด debug
