import csv
import re
import cv2
import numpy as np
import re
import easyocr
import re
from email.message import EmailMessage
import ssl
import smtplib
import threading
from PIL import Image, ImageDraw, ImageFont
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


font_path = "./static/NotoSansJP-Regular.ttf"  # Thay bằng đường dẫn font phù hợp
font = ImageFont.truetype(font_path, size=40)

# SAVE NUMBER PLATE TO CSV
def save_plate_to_csv(filename, plate, frame_count):
    """
    Lưu biển số và số khung hình liên tiếp vào file CSV.
    """
    
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([plate, frame_count])
    print(f"Lưu biển số: {plate} - {frame_count} khung hình")

def upscale_image(image, scale=2):
    t_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = t_gray.shape
    scale = 500 / w
    return cv2.resize(t_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def rotate_plate(image):
    # t_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t_blur = cv2.GaussianBlur(image, (3, 3), 0)
    t_thresh = cv2.threshold(t_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - t_thresh
    contours, _ = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    center, size, angle = rect
    if angle > 45:
        angle = angle - 90
        
    (h, w) = image.shape[:2]  # Lấy chiều cao, chiều rộng ảnh
    center = (w // 2, h // 2)  # Tâm ảnh
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Tạo ma trận xoay
    rotated_image = cv2.warpAffine(invert, M, (w, h))
    
    # Biến đổi tọa độ các góc
    rotated_box = []
    for point in box:
        x, y = point
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rotated_box.append((new_x, new_y))

    rotated_box = np.array(rotated_box, dtype=np.int32)

    # calPx = 0
    # Tính khung mới và cắt
    x_min = max(0, np.min(rotated_box[:, 0]))
    y_min = max(0, np.min(rotated_box[:, 1]))
    x_max = min(w, np.max(rotated_box[:, 0]))
    y_max = min(h, np.max(rotated_box[:, 1]))

    aligned = rotated_image[int(y_min):int(y_max), int(x_min):int(x_max)]
    
    def white_percentage(arr):
        return np.sum(arr > 100) / len(arr)

    # Định nghĩa ngưỡng tỷ lệ trắng
    threshold = 0.8

    # Lặp qua các biên để kiểm tra tỷ lệ trắng và thu hẹp dần
    height, width = aligned.shape
    top, bottom, left, right = 0, height - 1, 0, width - 1

    # Kiểm tra các biên trên (top) và dưới (bottom)
    while white_percentage(aligned[top, :]) < threshold and top < height // 2:
        aligned[top, :] = 255
        top += 1  # Dịch lên từng chút
        
    while white_percentage(aligned[bottom, :]) < threshold and bottom > height // 2:
        aligned[bottom, :] = 255
        bottom -= 1  # Dịch xuống từng chút

    # Kiểm tra các biên trái (left) và phải (right)
    while white_percentage(aligned[:, left]) < threshold and left < width // 2:
        aligned[:, left] = 255
        left += 1  # Dịch sang phải từng chút
        
    while white_percentage(aligned[:, right]) < threshold and right > width // 2:
        aligned[:, right] = 255
        right -= 1  # Dịch sang trái từng chút

    # last_img = aligned
    last_img = cv2.GaussianBlur(aligned, (3, 3), 0)
    
    return  last_img

def handle_number_plate(numberPlate, image_name, vehicles):
    # Mẫu regex kiểm tra xe máy và xe ô tô
    moto_regex = r'(\d{2}[A-Z]\d{1})(\d{4,5})'  # Xe máy: 11A11111 hoặc 11A111111
    car_regex = r'(\d{2}[A-Z])(\d{4,5})'         # Xe ô tô: 11A1111 hoặc 11A11111
    
    match = ''
    vehicleType = 2
    for key, value in vehicles.items():
        if value.lower() in image_name.lower():  # So sánh không phân biệt hoa/thường
            vehicleType = key
            break
        
    # Tìm phần biển số hợp lệ
    if vehicleType == 3:
        match = re.search(moto_regex, numberPlate)
    else:
        match = re.search(car_regex, numberPlate)

    if match:
        part1, part2 = match.groups()
        if len(part2) == 4:
            return f"{part1}-{part2}"  # Xe ô tô định dạng 11A-1111
        elif len(part2) == 5:
            return f"{part1}-{part2[:3]}.{part2[3:]}"  # Xe ô tô định dạng 11A-111.11
        
    # Trường hợp không hợp lệ
    return "???"

def get_number_plate_jp(image):
    h, w, _ = image.shape
    scale = 112 / w
    img_r =  cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    reader = easyocr.Reader(['ja'])

    results = reader.readtext(img_r)
    number = ''
    for detection in results:
        _, text, scoreT = detection
        number += re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFFA-Z0-9]', '', text.upper())
    
    pattern = r"[\u4E00-\u9FFF]{2}\d{3}[\u3040-\u309F]\d{3,4}"

    # Tìm tất cả các phần khớp với định dạng
    matches = re.findall(pattern, number)

    # In các kết quả
    if matches:
        for match in matches:
            return match
        
    return "???"

def recognize_license_plate(image, image_name, vehicles):
    h, w = image.shape[:2]
    if h < w / 2:
        text = pytesseract.image_to_string(image[:, 10:], lang='eng', config=f'--psm 6')
    else:
        text = pytesseract.image_to_string(image[:h//2, 30:w-30], lang='eng', config=f'--psm 6')
        text += pytesseract.image_to_string(image[h//2:, :], lang='eng', config=f'--psm 6')
    valid_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
    text_format = handle_number_plate(valid_text, image_name, vehicles)
    return text_format

# DRAW RECTANGLE AND NUMBER FOR LIENSCE PLATE
def draw_plate(image, plate_detail, number, time):
    if number == '???':
        return image
    
    x1, y1, x2, y2, score, class_id = plate_detail
    
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    draw = ImageDraw.Draw(pil_image)
    position = (int(x1), int(y1))
    color = (255, 0, 0)
    draw.text(position, number, font=font, fill=color)

    frame = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    
    return frame_bgr
    # cv2.putText(image, f'{number} | {time}', (int(x1), int(y1 - scale / 64)), cv2.FONT_HERSHEY_SIMPLEX, scale / 640, (255, 0, 0), scale // 240)

# ===== SEND EMAIL =====
def send_email(numberPlate, img, time):
    EMAIL_SENDER = 'leetuan0388@gmail.com'
    EMAIL_PASSWORD = 'riff jfty cgjc cars'
    EMAIL_RECEIVER = ['letuan2k1125@gmail.com', 'letuan20001125@gmail.com']
    # EMAIL_RECEIVER = ['letuan2k1125@gmail.com', 'giant.killng.no.1@gmail.com', 'shigeo_tomori0952176@lsi-dev.co.jp']
    
    subject='ナンバープレート情報'

    b_contact = """
    レ・トゥアン
    leetuan0388@gmail.com
    """
    
    b_infor = f"""
    - **ナンバープレート**: [{numberPlate}]
    - **発見時刻**: [{time}]
    - **カメラの位置**: [ラズベリーカメラ]
    """
        
    body = f"""
    親愛なる、

    当社のカメラシステムがナンバープレートの番号を検出しました。詳細は以下の通りです。
    {b_infor}
    ご質問がある場合やさらにサポートが必要な場合は、弊社の技術部門にお問い合わせください。

    よろしくお願いします、
    {b_contact}
    """
    print(b_infor)
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = EMAIL_RECEIVER
    em['Subject'] = subject
    em.set_content(body)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_data = buffer.tobytes()
    
    
    em.add_attachment(img_data, maintype='image', subtype='jpeg', filename='number_plate.jpg')

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(em)
        # smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, em.as_string())  
        
def send_email_async(numberPlate, img, time):
    threading.Thread(target=send_email, args=(numberPlate, img, time)).start()

        
        