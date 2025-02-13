import cv2
import time
import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime
from keras_facenet import FaceNet
import mediapipe as mp
from utils import send_email_async, upscale_image, rotate_plate, recognize_license_plate, draw_plate, get_number_plate_jp

# Đặt URL của video feed từ Raspberry Pi
# url = "http://192.168.20.81:5000/stream"
url = "http://10.242.7.129:5000/stream"
try:
    # Mở luồng video
    cap = cv2.VideoCapture(0)

    last_time = time.time()
    frame_count = 0
    fps = 0

    mode = 0
    type = 'jp'

    # 1. Vehicle
    coco_model = YOLO('./models/yolo11n.pt')
    vehicles = {2: 'Car', 3: 'motorcycle', 5: 'bus', 7: 'Truck'}

    # 2. Face
    facenet = FaceNet()
    face_embeddings = np.load("./models/faces_embeddings_done_4classes.npz")
    Y = face_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    model = pickle.load(open("./models/svm_model_160x160.pkl", 'rb'))
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # 3. Plate number
    # license_plate_detector = YOLO('./models/license_plate_detector.pt')
    model_path = './#test/models/license_plate_detector_saved_model/license_plate_detector_float16.tflite'
    license_plate_detector = YOLO(model_path, task="detect")
    plate_states = {}
    CHECK_INTERVAL = 15
    cascade= cv2.CascadeClassifier("./models/haarcascade_russian_plate_number.xml")

    if not cap.isOpened():
        print("Không thể kết nối tới luồng video.")
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_cp = frame.copy()
        
        scale, img_w = frame.shape[:2]
        if scale < img_w:
            scale = img_w
        
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
        day = now.day  # Lấy ngày
        hour = now.hour  # Lấy giờ
        current_time = f"{day}th:{hour}h"
        
        # ===== Vehicles Detection ===== #
        if mode == 1:
            detections = coco_model(frame)[0]
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and score > 0.4:
                    score = round(score, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), scale // 240)
                    cv2.putText(frame, f'{vehicles[class_id]} | {current_time}', (int(x1), int(y1 - scale / 64)), cv2.FONT_HERSHEY_SIMPLEX, scale / 640, (255, 0, 0), scale // 240)
        
        # ===== Face Recognition ===== #
        if mode == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Get face vector
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Cut face image
                    img = rgb_frame[y:y+h, x:x+w]
                    
                    try:
                        img = cv2.resize(img, (160,160)) # Resize 160x160
                    except Exception as e:
                        continue
                    
                    img = np.expand_dims(img,axis=0)
                    test_im_embedding = facenet.embeddings(img)
                    
                    y_probs = model.predict_proba([test_im_embedding[0]])
                    max_prob = np.max(y_probs)
                    ypred = np.argmax(y_probs)
        
                    encoder = LabelEncoder()
                    encoder.fit(Y)
                    
                    if max_prob < 0.7:
                        final_name = "Unknown"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f'{final_name} | {current_time}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        final_name = encoder.inverse_transform([ypred])[0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f'{final_name} | {current_time}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame, f'{final_name} | {current_time}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ===== Number plates Recognition ===== #
        if mode == 3:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            nplate=cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in nplate:
                wT,hT,cT=frame.shape
                a,b=(int(0.02*wT),int(0.02*hT))
                # plate=img[y+a:y+h-a,x+b:x+w-b,:]

                cv2.rectangle(frame, (x+b, y+a), (x+w-b, y+h-a), (0, 255, 0), 2)

            # license_plates = license_plate_detector(frame)[0]
            # for license_plate in license_plates.boxes.data.tolist():
            #     x1, y1, x2, y2, score, class_id = license_plate
                
            #     if score > 0.6:
            #         plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]
            #         plate_upscale = upscale_image(plate_crop)
                    
            #         if type == 'jp':
            #             plate_number = get_number_plate_jp(plate_crop)
                        
            #         if type == 'vn':
            #             plate_rotate = rotate_plate(plate_upscale)
            #             plate_number = recognize_license_plate(plate_crop, 'car', vehicles)
                    
            #         frame = draw_plate(frame, license_plate, plate_number, current_time)
                    
            #         # Set up for send mail
            #         if plate_number != '???':
            #             if plate_number not in plate_states:
            #                 plate_states[plate_number] = {'count': 0, 'last': None, 'sent': False}
                        
            #             plate_states[plate_number]['count'] += 1
            #             plate_states[plate_number]['last'] = formatted_time
                        
            #             if plate_states[plate_number]['count'] == 5:
            #                 # send_email_async(plate_number, frame_cp, formatted_time)
            #                 print("================Send email===================")
            
            # if now.second % CHECK_INTERVAL == 0:
                
            #     for plate in plate_states:
            #         if (current_time - plate_states[plate]['last']).total_seconds() > CHECK_INTERVAL:
            #             del plate_states[plate]
            #     print(plate_states)
            #     plate_states = send_email_async(plate_states)
            
        # Update FPS every 1s
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1:
            fps = frame_count
            frame_count = 0  
            last_time = current_time  

        # Show the FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Video from Raspberry Pi', frame)
        
        # Check keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('0'):
            mode = 0
        if key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('j'):
            type = 'jp'
        elif key == ord('v'):
            type = 'vn'
        elif key == ord('q') or key == ord('Q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)
                        