from ultralytics import YOLO
import cv2
import time 

# Load mô hình YOLO-Pose
model = YOLO("./models/yolo11n-pose.pt")  # Hoặc "yolov8s-pose.pt" để có độ chính xác cao hơn

# Mở camera
cap = cv2.VideoCapture(0)

last_time = time.time()
frame_count = 0
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán Pose
    results = model(frame)

    # Vẽ keypoints lên ảnh
    frame_annotated = results[0].plot()

    # Update FPS every 1s
    frame_count += 1
    current_time = time.time()
    if current_time - last_time >= 1:
        fps = frame_count
        frame_count = 0  
        last_time = current_time  

    # Show the FPS 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_annotated, f"FPS: {fps}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Hiển thị kết quả
    cv2.imshow("YOLOv11 Pose Estimation", frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




### =====================================================================
# import subprocess
# import cv2
# import numpy as np

# def run_gstreamer():
#     gstreamer_command = r'C:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! appsink'
    
#     # Chạy lệnh GStreamer qua subprocess
#     process = subprocess.Popen(gstreamer_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    

#     while True:
#         # Đọc dữ liệu video từ stdout của GStreamer
#         raw_frame = process.stdout.read(1024)  # Giả sử độ phân giải 640x480 và 3 kênh màu (RGB)

#         if not raw_frame:
#             break  # Nếu không còn dữ liệu, thoát khỏi vòng lặp

#         # Chuyển dữ liệu byte thành mảng NumPy (khung hình OpenCV)
#         # frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((480, 640, 3))  # Chuyển thành dạng 480x640

#         # Hiển thị khung hình sử dụng OpenCV
#         cv2.imshow("Video Stream", raw_frame)

#         # Kiểm tra xem người dùng có nhấn 'q' để thoát không
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Đóng tất cả cửa sổ và kết thúc
#     process.stdout.close()
#     process.stderr.close()
#     process.wait()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_gstreamer()


# plate_states = {}
# for plate in plate_states:
#     print("1")
# plate_number = '1234'
# # Set up for send mail
# if plate_number != '???':
#     if plate_number not in plate_states:
#         plate_states[plate_number] = {'count': 0, 'last': None, 'sent': False}
    
#     plate_states[plate_number]['count'] += 1
#     plate_states[plate_number]['last'] = 2

# print(plate_states)
# if plate_states[plate_number]['count'] >= 1:
#     print("OK")
# import cv2

# # Chọn camera (0 là camera tích hợp, 1 là camera rời)
# camera_index = 1  # Thay đổi nếu cần thiết

# # Mở camera
# cap = cv2.VideoCapture(camera_index)

# if not cap.isOpened():
#     print(f"Không thể mở camera {camera_index}")
#     exit()

# print(f"Đang sử dụng camera {camera_index}")
# while True:
#     # Đọc frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể nhận frame từ camera.")
#         break

#     # Hiển thị frame
#     cv2.imshow(f"Camera {camera_index}", frame)

#     # Nhấn phím 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng camera và đóng cửa sổ
# cap.release()
# cv2.destroyAllWindows()


### =====================================================================
# from PIL import Image, ImageDraw, ImageFont
# import cv2
# import numpy as np

# import time
# from ultralytics import YOLO
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from datetime import datetime
# from keras_facenet import FaceNet
# import mediapipe as mp
# from utils import send_email_async, upscale_image, rotate_plate, recognize_license_plate, draw_plate, get_number_plate_jp

# # 1. Vehicle
# coco_model = YOLO('./models/yolo11n.pt')
# vehicles = {2: 'Car', 3: 'motorcycle', 5: 'bus', 7: 'Truck'}

# # 2. Face
# facenet = FaceNet()
# face_embeddings = np.load("./models/faces_embeddings_done_4classes.npz")
# Y = face_embeddings['arr_1']
# encoder = LabelEncoder()
# encoder.fit(Y)
# model = pickle.load(open("./models/svm_model_160x160.pkl", 'rb'))
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# # Đường dẫn đến file font (font hỗ trợ tiếng Nhật, như Noto Sans JP, Meiryo, v.v.)
# font_path = "./static/NotoSansJP-Regular.ttf"  # Thay bằng đường dẫn font phù hợp
# font = ImageFont.truetype(font_path, size=40)

# # Khởi tạo camera
# url = "http://10.242.7.129:5000/stream"

# # Mở luồng video
# cap = cv2.VideoCapture(0)  # Nếu bạn dùng camera khác, đổi 0 thành index camera

# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Chuyển ảnh từ BGR sang RGB để sử dụng với Pillow
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(frame_rgb)

#     # Vẽ văn bản tiếng Nhật lên khung hình
#     draw = ImageDraw.Draw(pil_image)
#     text = "こんにちは、世界！"  # Văn bản tiếng Nhật
#     position = (50, 50)  # Tọa độ hiển thị
#     color = (255, 0, 0)  # Màu chữ (RGB)
#     draw.text(position, text, font=font, fill=color)

#     # Chuyển đổi lại từ Pillow (RGB) sang OpenCV (BGR)
#     frame = np.array(pil_image)
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
#     cv2.rectangle(frame_bgr, (int(50), int(50)), (int(200), int(200)), (255, 0, 0), 2)

#     # Hiển thị khung hình
#     cv2.imshow("Real-Time Camera", frame_bgr)

#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()
