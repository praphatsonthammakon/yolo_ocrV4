#! D:\test_env_ai\myenv_ocrYolo\Scripts\python.exe

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import easyocr
import re
import os
import streamlit as st

# ตั้งค่า EasyOCR
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=os.getcwd().replace("\\", "//"), download_enabled=False)

# โหลดโมเดล YOLO
model = YOLO("best_ticket_v2.pt")

# สร้างอินเตอร์เฟซใน Streamlit
st.text("Stream IoT Solution")
st.title("Reading Parking Tickets Using OCR")

if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

files = st.file_uploader(
    "Upload some files",
    type=["jpg", "jpeg", "png"],
    key=st.session_state["file_uploader_key"],
)



if files:
    st.session_state["uploaded_files"] = files
    # แสดง progress bar
    progress = st.empty()
    progress = st.progress(0)
    
    # โหลดและแปลงภาพ
    image = Image.open(files)
    image_cv22 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # คำนวณความสูงและความกว้างของภาพต้นฉบับ
    height, width = image_cv22.shape[:2]

    # คำนวณอัตราส่วนของภาพต้นฉบับ
    proportion = width / height

    # ตั้งค่าความกว้างใหม่ที่ต้องการ
    new_width = 1400

    # คำนวณความสูงใหม่ให้สัมพันธ์กันตามอัตราส่วน
    new_height = int(new_width / proportion)

    # ปรับขนาดภาพ
    image_cv2 = cv2.resize(image_cv22, (new_width, new_height))
    
    # หมุนภาพ 90 องศา
    image_cv2 = cv2.rotate(image_cv2, cv2.ROTATE_90_CLOCKWISE)

    # สร้าง placeholder สำหรับการแสดงภาพ
    image_placeholder = st.empty()
    image_placeholder.image(image_cv2, channels="BGR", caption="Uploaded Image")

    progress.progress(10)  # อัปเดต progress bar

    # ทำการพยากรณ์โดยใช้โมเดล
    results = model(image_cv2)
    progress.progress(40)  # อัปเดต progress bar

    # หาค่าที่มี confidence สูงสุด
    best_box = None
    best_confidence = 0

    for result in results:
        for box in result.boxes:
            if box.conf[0] > best_confidence:  # เปรียบเทียบค่า confidence
                best_confidence = box.conf[0]
                best_box = box.xyxy[0]
    progress.progress(60)  # อัปเดต progress bar

    # ฟังก์ชันตรวจสอบเงื่อนไข
    def filter_text(text):
        text = text.strip()
        if re.search(r'\d{5}', text):
            return True
        if re.search(r'\d{4}', text):
            return True
        if text.count('/') == 2:
            return True
        if ':' in text:
            return True
        return False

    # ถ้าพบกรอบที่ดีที่สุด
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)  # ดึงค่าที่ถูกต้องจาก numpy array และแปลงเป็น int
        cropped_image = image_cv2[y1:y2, x1:x2]  # ครอบภาพตามกรอบที่ตรวจจับได้
        progress.progress(70)  # อัปเดต progress bar

        # อัปเดตภาพที่ครอบไว้ใน placeholder
        image_placeholder.image(cropped_image, channels="BGR", caption="Cropped Image")

        # เริ่มกระบวนการ OCR ตรงนี้
        results_ocr = reader.readtext(cropped_image)
        progress.progress(90)  # อัปเดต progress bar

        # กรองข้อความตามเงื่อนไขที่กำหนด
        filtered_ocr = [result[1].strip().replace(" ", "") for result in results_ocr if filter_text(result[1].replace(" ", ""))]

        # แสดงผลข้อความที่กรองแล้วใน Streamlit
        # with image_placeholder.container():
        try:
            if len(filtered_ocr) >= 4:
                st.text("หมายเลขบัตรขาว : " + filtered_ocr[0])
                st.text("หมายเลขทะเบียน : " + filtered_ocr[1])
                st.text("วันที่           : " + filtered_ocr[2])
                st.text("เวลา           : " + filtered_ocr[3])

                progress.progress(100)  # อัปเดต progress bar เป็น 100%
                st.success("สามารถอ่านข้อมูลได้ครบถ้วนจากภาพที่อัปโหลด")

            else:
                st.error("ไม่สามารถอ่านข้อมูลได้ครบถ้วนจากภาพที่อัปโหลด")

        except IndexError:
            st.error("ไม่สามารถอ่านข้อมูลได้ครบถ้วนจากภาพที่อัปโหลด")
 
    else:
        st.error("ไม่สามารถตรวจจับกรอบที่มีค่า confidence สูงสุดได้")  




#st.write("Uploaded files:", st.session_state["uploaded_files"])
