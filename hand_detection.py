import cv2  # นำเข้า OpenCV สำหรับการจัดการวิดีโอและภาพ
import mediapipe as mp  # นำเข้า MediaPipe สำหรับการตรวจจับมือ

class HandDetector:  # สร้างคลาส HandDetector สำหรับตรวจจับมือ
    def __init__(self, mode=False, hands_to_track=2,
                 detector_confidence=0.5, tracking_confidence=0.5) -> None:
        # ฟังก์ชันเริ่มต้นของคลาส ใช้กำหนดค่าพารามิเตอร์ต่าง ๆ สำหรับการตรวจจับมือ
        self.mode = mode  # โหมดการตรวจจับภาพแบบนิ่ง
        self.hands_to_track = hands_to_track  # จำนวนมือที่ต้องการติดตาม
        self.detection_confidence = detector_confidence  # ความมั่นใจในการตรวจจับ
        self.tracking_confidence = tracking_confidence  # ความมั่นใจในการติดตาม
        
        self.np_hand = mp.solutions.hands  # นำเข้าโมดูลการตรวจจับมือจาก MediaPipe
        self.hands = self.np_hand.Hands(  # สร้างอ็อบเจ็กต์ Hands สำหรับการตรวจจับมือ
            static_image_mode=self.mode,  # กำหนดให้ตรวจจับในโหมดภาพนิ่งหรือไม่
            max_num_hands=self.hands_to_track,  # กำหนดจำนวนมือสูงสุดที่จะตรวจจับ
            min_detection_confidence=self.detection_confidence,  # กำหนดความมั่นใจขั้นต่ำในการตรวจจับ
            min_tracking_confidence=self.tracking_confidence  # กำหนดความมั่นใจขั้นต่ำในการติดตาม
        )

        self.mp_drawing_utils = mp.solutions.drawing_utils  # นำเข้าฟังก์ชันวาดจาก MediaPipe
        self.mp_drawing_styles = mp.solutions.drawing_styles  # นำเข้าสตีลการวาดจาก MediaPipe

        self.tips_ids = [4, 8, 12, 16, 20]  # รหัสของจุดปลายนิ้ว (tips) ที่จะตรวจจับ

    def find_hands(self, img, draw=True):  # ฟังก์ชันสำหรับตรวจจับมือในภาพ
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB
        self.result = self.hands.process(img_rgb)  # ประมวลผลภาพเพื่อหามือ

        if self.result.multi_hand_landmarks:  # ถ้าพบมือในภาพ
            if draw:  # ถ้าต้องการวาดภาพ
                for hand_landmark in self.result.multi_hand_landmarks:  # สำหรับแต่ละมือที่ตรวจจับได้
                    self.mp_drawing_utils.draw_landmarks(  # วาดจุดบนมือ
                        img,
                        hand_landmark,
                        self.np_hand.HAND_CONNECTIONS,  # เชื่อมต่อจุดในมือ
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),  # สไตล์การวาดจุดมือ
                        self.mp_drawing_styles.get_default_hand_connections_style()  # สไตล์การวาดการเชื่อมต่อ
                    )
        return img  # คืนค่าภาพที่มีการวาดจุดมือ

    def find_position(self, img, hand_index=0, draw=True):  # ฟังก์ชันค้นหาตำแหน่งของจุดบนมือ
        self.land_mark_list = []  # รายการเก็บตำแหน่งจุดบนมือ

        if self.result.multi_hand_landmarks:  # ถ้าพบมือในภาพ
            interes_hand = self.result.multi_hand_landmarks[hand_index]  # เลือกมือที่ต้องการตรวจจับ

            for id, landmark in enumerate(interes_hand.landmark):  # สำหรับแต่ละจุดบนมือ
                h, w, c = img.shape  # รับขนาดของภาพ (สูง, กว้าง, ช่องสี)

                cx, cy = int(landmark.x * w), int(landmark.y * h)  # คำนวณตำแหน่ง (cx, cy) ของจุดบนภาพ

                self.land_mark_list.append([id, cx, cy])  # เพิ่มตำแหน่งจุดลงในรายการ

                if draw:  # ถ้าต้องการวาด
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # วาดวงกลมที่จุดบนมือ

        return self.land_mark_list  # คืนค่ารายการตำแหน่งจุดบนมือ

    def fingers_up(self):  # ฟังก์ชันตรวจสอบว่านิ้วไหนยกขึ้น
        if len(self.land_mark_list) != 0:  # ถ้ามีการตรวจจับจุดบนมือ
            fingers_counter = []  # รายการสำหรับนับจำนวนที่ยกนิ้ว

            for tip_id in range(1, 5):  # สำหรับนิ้ว 1 ถึง 4 (นิ้วชี้ถึงนิ้วก้อย)
                if self.land_mark_list[self.tips_ids[tip_id]][2] < self.land_mark_list[self.tips_ids[tip_id] - 2][2]:
                    fingers_counter.append(1)  # ถ้านิ้วยกขึ้น เพิ่ม 1
                else:
                    fingers_counter.append(0)  # ถ้านิ้วยังอยู่ในตำแหน่งปกติ เพิ่ม 0

            # ตรวจสอบนิ้วโป้ง
            if self.land_mark_list[self.tips_ids[0]][1] > self.land_mark_list[self.tips_ids[0] - 1][1]:
                fingers_counter.append(1)  # ถ้านิ้วโป้งยกขึ้น เพิ่ม 1
            else:
                fingers_counter.append(0)  # ถ้านิ้วโป้งไม่ยกขึ้น เพิ่ม 0

            return fingers_counter  # คืนค่ารายการจำนวนที่ยกนิ้ว
            
        return None  # ถ้าไม่มีจุดบนมือ คืนค่า None
