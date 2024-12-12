import os
import cv2
import torch 
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Mengimpor SAHI untuk object detection
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict

app = Flask(__name__)

# Konfigurasi direktori upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan direktori upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model_yolov5s = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/yolov5s_students.pt')
model_yolov5kd = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/kd_yolov5s.pt')
model_sahi = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path='./models/kd_yolov5s.pt',
    confidence_threshold=0.3
)

# Definisi nama kelas VisDrone (digunakan di beberapa fungsi)
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Definisikan warna untuk setiap kelas (digunakan di beberapa fungsi)
COLORS = [
    (4, 42, 255),    
    (11, 219, 235),    
    (243, 243, 243),    
    (0, 223, 183), 
    (17, 31, 104),  
    (255, 111, 221),  
    (255, 68, 79),    
    (204, 237, 0),    
    (0, 243, 68),   
    (189, 0, 255)    
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detection(image, bbox, cls_index, confidence):
    """
    Menggambar deteksi pada gambar
    """
    # Pastikan indeks kelas valid
    if cls_index < 0 or cls_index >= len(CLASS_NAMES):
        return image
    
    # Konversi koordinat
    xyxy = [int(x) for x in bbox]
    
    # Pilih warna berdasarkan kelas
    color = COLORS[cls_index]
    
    # Buat label
    label = f'{CLASS_NAMES[cls_index]} {confidence:.2f}'
    
    # Gambar bounding box
    cv2.rectangle(
        image, 
        (xyxy[0], xyxy[1]), 
        (xyxy[2], xyxy[3]), 
        color, 
        2
    )
    
    # Tambahkan label dengan background warna
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(
        image, 
        (xyxy[0], xyxy[1] - h - 10), 
        (xyxy[0] + w, xyxy[1]), 
        color, 
        -1
    )
    
    # Tulis teks label
    cv2.putText(
        image, 
        label, 
        (xyxy[0], xyxy[1] - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255),  # Warna teks putih
        1
    )
    
    return image

def process_image_yolov5s(image_path):
    # Baca gambar asli
    image = cv2.imread(image_path)
    
    # Deteksi objek
    results = model_yolov5s(image_path)
    
    # Proses setiap deteksi
    for *xyxy, conf, cls in results.xyxy[0]:
        image = draw_detection(image, xyxy, int(cls), conf)
    
    return image

def process_image_yolov5kd(image_path):
    # Baca gambar asli
    image = cv2.imread(image_path)
    
    # Deteksi objek
    results = model_yolov5kd(image_path)
    
    # Proses setiap deteksi
    for *xyxy, conf, cls in results.xyxy[0]:
        image = draw_detection(image, xyxy, int(cls), conf)
    
    return image

def process_image_sahi(image_path):
    # Deteksi menggunakan SAHI
    results = get_sliced_prediction(
        image_path, 
        model_sahi, 
        slice_height=256, 
        slice_width=256, 
        overlap_height_ratio=0.2, 
        overlap_width_ratio=0.2
    )
    
    # Konversi hasil ke gambar dengan bounding box dan label
    image = cv2.imread(image_path)
    
    for prediction in results.object_prediction_list:
        bbox = prediction.bbox.to_xyxy()
        confidence = prediction.score.value
        cls_index = prediction.category.id
        
        image = draw_detection(image, bbox, cls_index, confidence)
    
    return image

def process_video_detection(video_path, detection_type):
    """
    Memproses video dengan model deteksi tertentu
    detection_type: 'yolov5s', 'yolov5kd', atau 'sahi'
    """
    cap = cv2.VideoCapture(video_path)
    
    # Tentukan model deteksi
    if detection_type == 'yolov5s':
        model = model_yolov5s
        detect_func = lambda frame: frame
    elif detection_type == 'yolov5kd':
        model = model_yolov5kd
        detect_func = lambda frame: frame
    elif detection_type == 'sahi':
        model = model_sahi
        detect_func = lambda frame: frame
    else:
        raise ValueError("Tipe deteksi tidak valid")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Deteksi objek bergantung pada tipe
        if detection_type in ['yolov5s', 'yolov5kd']:
            results = model(frame)
            for *xyxy, conf, cls in results.xyxy[0]:
                frame = draw_detection(frame, xyxy, int(cls), conf)
        elif detection_type == 'sahi':
            # Konversi frame ke file sementara
            temp_frame_path = os.path.join(UPLOAD_FOLDER, 'temp_frame.jpg')
            cv2.imwrite(temp_frame_path, frame)
            
            # Deteksi dengan SAHI
            results = get_sliced_prediction(
                temp_frame_path, 
                model_sahi, 
                slice_height=256, 
                slice_width=256, 
                overlap_height_ratio=0.2, 
                overlap_width_ratio=0.2
            )
            
            # Gambar deteksi
            for prediction in results.object_prediction_list:
                bbox = prediction.bbox.to_xyxy()
                confidence = prediction.score.value
                cls_index = prediction.category.id
                
                frame = draw_detection(frame, bbox, cls_index, confidence)
            
            # Hapus file sementara
            os.remove(temp_frame_path)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Proses deteksi untuk 3 model
            result_yolov5s = process_image_yolov5s(filepath)
            result_yolov5kd = process_image_yolov5kd(filepath)
            result_sahi = process_image_sahi(filepath)
            
            # Konversi hasil ke format yang dapat ditampilkan
            _, buffer_yolov5s = cv2.imencode('.jpg', result_yolov5s)
            _, buffer_yolov5kd = cv2.imencode('.jpg', result_yolov5kd)
            _, buffer_sahi = cv2.imencode('.jpg', result_sahi)
            
            # Encode ke base64 untuk ditampilkan di HTML
            import base64
            img_yolov5s = base64.b64encode(buffer_yolov5s).decode('utf-8')
            img_yolov5kd = base64.b64encode(buffer_yolov5kd).decode('utf-8')
            img_sahi = base64.b64encode(buffer_sahi).decode('utf-8')
            
            return render_template('image_detection.html', 
                                   img_yolov5s=img_yolov5s, 
                                   img_yolov5kd=img_yolov5kd, 
                                   img_sahi=img_sahi)
    
    return render_template('image_detection.html')

@app.route('/video_detection', methods=['GET', 'POST'])
def video_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return render_template('video_detection.html', filename=filename)
    
    return render_template('video_detection.html')

@app.route('/video_feed/<detection_type>/<filename>')
def video_feed(detection_type, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(process_video_detection(filepath, detection_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html', class_names=CLASS_NAMES)

if __name__ == '__main__':
    app.run(debug=True)