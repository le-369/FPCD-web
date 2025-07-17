import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from datetime import timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.utils.image_utils import img_to_array
from utils import parse_result, isFakePlate
import pandas as pd
import tempfile
import shutil
import tensorflow as tf
# from torchvision import transforms
from ultralytics import YOLO
from typing import cast, Optional, Tuple
import sys
import json
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont
import os

# 确保当前目录在 Python 路径中，以便正确解析本地包
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from inference.predict_car_type import predict_vehicle_type, VEHICLE_HEIGHT, VEHICLE_WIDTH
from inference.predict_car_brand import load_models as load_brand_models, predict_two_stage
from inference.license_plate.plate_recognition.plate_rec import get_plate_result,init_model as plate_init_model
from inference.license_plate.plate_recognition.double_plate_split_merge import get_split_merge

if not hasattr(np, 'int'):
    np.int = int  # type: ignore

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 模型路径
YOLO_MODEL_PATH = os.path.join(current_dir,"weights", "yolo_car_detector.pt")
BRAND_MODEL_PATH = os.path.join(current_dir, "weights", "car_brand_classifier.pth")
LABEL_MAP_PATH = os.path.join(current_dir, "weights", "label_map.json")
TYPE_MODEL_PATH = os.path.join(current_dir, "weights", "vehicle_type.hdf5")
# ===== 颜色模型常量 =====
YOLO_COLOR_MODEL_PATH = os.path.join(current_dir, "weights", "yolo_car_detector.pt")
COLOR_MODEL_PATH      = os.path.join(current_dir, "weights", "vehicle_color.hdf5")
COLOR_LABELS = ["black","blue","brown","green","red","silver","white","yellow"]
# ===== 车牌检测与识别模型路径 =====
PLATE_DET_MODEL_PATH  = os.path.join(current_dir, "weights", "yolov8s.pt")
PLATE_REC_MODEL_PATH  = os.path.join(current_dir, "weights", "plate_rec_color.pth")
# 安全带模型路径
CAR_DET_MODEL_PATH = os.path.join(current_dir, 'weights', 'yolo_car_detector.pt')
BELT_DET_MODEL_PATH = os.path.join(current_dir, 'weights', 'yolo_car_belt_detector.pt')
# Flask应用配置
app = Flask(__name__)
# 设置session密钥
app.secret_key = 'your_very_secret_key' # 在生产环境中请使用更安全的密钥
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# 全局变量
vehicle_info_database = None
net = None
meta = None
v_color_model = None
v_type_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = None
brand_model = None
type_model = None
idx_to_brand = None
temp_dir = None
# ===== 新增全局变量 =====
yolo_color_model = None  # YOLO for color detection
color_model      = None  # Keras color classifier
# ===== 车牌全局模型 =====
plate_det_model  = None
plate_rec_model  = None
# 安全带模型
car_model = None
belt_model = None

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # robust path for font file
    font_path = r"inference\license_plate\plate_recognition\platech.ttf"
    fontText = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def allowed_file(filename: Optional[str]) -> bool:
    """检查文件类型是否允许"""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def load_models():
    """加载所有模型"""
    global net, meta, v_color_model, v_type_model
    try:
        # 检查文件是否存在
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO模型文件不存在: {YOLO_MODEL_PATH}")
        if not os.path.exists(BRAND_MODEL_PATH):
            raise FileNotFoundError(f"品牌分类模型文件不存在: {BRAND_MODEL_PATH}")
        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"标签映射文件不存在: {LABEL_MAP_PATH}")
        if not os.path.exists(TYPE_MODEL_PATH):
            raise FileNotFoundError(f"类型模型文件不存在: {TYPE_MODEL_PATH}")
        
        # 加载模型
        # net = dn.load_net(YOLO_MODEL_PATH.encode('utf-8'), BRAND_MODEL_PATH.encode('utf-8'), 0)
        # meta = dn.load_meta(LABEL_MAP_PATH.encode('utf-8'))
        v_color_model = load_model(TYPE_MODEL_PATH)
        v_type_model = load_model(TYPE_MODEL_PATH)
        
        print("所有模型加载成功")
        return True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

def prepare_service():
    """准备服务，加载数据库"""
    global vehicle_info_database
    try:
        if not os.path.exists('vehicle-database.csv'):
            raise FileNotFoundError("车辆数据库文件不存在: vehicle-database.csv")
        vehicle_info_database = pd.read_csv('vehicle-database.csv')
        print("车辆数据库加载成功")
        return vehicle_info_database
    except Exception as e:
        print(f"数据库加载失败: {str(e)}")
        return None

def load_all_models():
    """加载所有模型"""
    global yolo_model, brand_model, type_model, yolo_color_model, color_model, plate_det_model, plate_rec_model, car_model, belt_model, idx_to_brand, temp_dir
    try:
        # 检查文件是否存在
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO模型文件不存在: {YOLO_MODEL_PATH}")
        if not os.path.exists(BRAND_MODEL_PATH):
            raise FileNotFoundError(f"品牌分类模型文件不存在: {BRAND_MODEL_PATH}")
        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"标签映射文件不存在: {LABEL_MAP_PATH}")
        if not os.path.exists(TYPE_MODEL_PATH):
            raise FileNotFoundError(f"类型模型文件不存在: {TYPE_MODEL_PATH}")
        if not os.path.exists(YOLO_COLOR_MODEL_PATH):
            raise FileNotFoundError(f"颜色YOLO模型不存在: {YOLO_COLOR_MODEL_PATH}")
        if not os.path.exists(COLOR_MODEL_PATH):
            raise FileNotFoundError(f"颜色分类模型不存在: {COLOR_MODEL_PATH}")
        if not os.path.exists(BELT_DET_MODEL_PATH):
            raise FileNotFoundError(f"安全带检测模型不存在: {BELT_DET_MODEL_PATH}")
        
        # 加载品牌分类模型
        yolo_model, brand_model, idx_to_brand = load_brand_models(
            YOLO_MODEL_PATH,
            BRAND_MODEL_PATH,
            LABEL_MAP_PATH,
            device
        )
        
        # 加载类型分类模型
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, 'vehicle_type.hdf5')
        shutil.copy2(TYPE_MODEL_PATH, temp_model_path)
        
        try:
            type_model = tf.keras.models.load_model(temp_model_path)
        except Exception as e:
            print(f"加载类型模型出错: {e}")
            type_model = tf.keras.models.load_model(temp_model_path, compile=False)
        
        # ===== 加载颜色相关模型 =====
        yolo_color_model = YOLO(YOLO_COLOR_MODEL_PATH)
        color_model      = tf.keras.models.load_model(COLOR_MODEL_PATH, compile=False)
        
        # ---- 加载车牌检测与识别模型 (使用现代YOLOv8接口) ----
        plate_det_model  = YOLO(PLATE_DET_MODEL_PATH)
        plate_rec_model  = plate_init_model(device, PLATE_REC_MODEL_PATH, is_color=True)
        
        # 加载安全带模型
        car_model = YOLO(CAR_DET_MODEL_PATH)
        belt_model = YOLO(BELT_DET_MODEL_PATH)
        
        print("所有模型加载成功")
        return True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def predict(image_path, vehicle_info_database):
    """预测函数：整合车型识别和品牌识别，并在图像上绘制检测框"""
    try:
        # 品牌识别
        pred_brand, pred_prob, image, bbox = predict_two_stage(
            image_path, yolo_model, brand_model, idx_to_brand, device
        )
        
        # 转换图像格式以便于OpenCV处理
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if bbox is None:
             # 如果未检测到车辆，也尝试检测车牌，并绘制车牌框
            plateNo, plate_rect = detect_plate_number_plate_module(image_cv)
            if plate_rect is not None:
                px1, py1, px2, py2 = plate_rect
                cv2.rectangle(image_cv, (px1, py1), (px2, py2), (255, 0, 0), 2) # 蓝色车牌框
            return "未识别", "未识别", "未识别", "未识别", "未检测到车辆", "安全带检测失败", image_cv

        # 将浮点坐标转换为整数
        x1, y1, x2, y2 = map(int, bbox)
        cropped_vehicle = image_cv[y1:y2, x1:x2]
        
        # 车型识别
        vehicle_type, _ = predict_vehicle_type(cropped_vehicle, type_model)

        # 颜色识别
        v_color = predict_vehicle_color(image_cv)

        # 车牌识别（使用 plate 模块，对整幅图）
        plateNo, plate_rect = detect_plate_number_plate_module(image_cv)

        # 安全带检测
        image_cv, belt_result = detect_seat_belt(image_cv)
        
        # --- 在图像上绘制检测框 ---
        # 绘制车辆检测框 (绿色)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 使用cv2ImgAddText绘制中文
        brand_text = pred_brand if isinstance(pred_brand, str) else ""
        image_cv = cv2ImgAddText(image_cv, brand_text, x1, y1 - 30, (0, 255, 0), 24)
        
        # 绘制车牌检测框 (蓝色)
        if plate_rect is not None:
            px1, py1, px2, py2 = plate_rect
            cv2.rectangle(image_cv, (px1, py1), (px2, py2), (255, 0, 0), 2)
            # 使用cv2ImgAddText绘制中文
            plate_text = plateNo if isinstance(plateNo, str) else ""
            image_cv = cv2ImgAddText(image_cv, plate_text, px1, py1 - 30, (255, 0, 0), 24)


        predictResult = "车牌未识别，无法判定，请重新上传"
        
        # 如果检测到品牌，进行套牌车检查
        if pred_brand and pred_brand != "未识别":
            inputCarInfo = [plateNo, pred_brand]
            isFake, true_car_brand = isFakePlate(inputCarInfo, vehicle_info_database)
            if isFake:
                predictResult = "这是一辆套牌车"
            else:
                predictResult = "这是一辆正常车"
        
        return plateNo, vehicle_type, v_color, pred_brand, predictResult, belt_result, image_cv
        
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        # 在出错时返回原始图片
        return "未识别", "未识别", "未识别", "未识别", "预测失败", "安全带检测失败", cv2.imread(image_path)

@app.route('/status')
def check_status():
    """渲染系统状态页面"""
    status = {
        "models_loaded": {
            "brand_model": brand_model is not None and yolo_model is not None,
            "type_model": type_model is not None,
            "color_model": color_model is not None and yolo_color_model is not None,
            "plate_model": plate_det_model is not None and plate_rec_model is not None,
            "belt_model": belt_model is not None,
        },
        "database_loaded": vehicle_info_database is not None,
    }
    
    all_models_loaded = all(status["models_loaded"].values())
    system_ready = all_models_loaded and status["database_loaded"]
    status["system_ready"] = system_ready
    
    return render_template('status.html', status=status)

@app.route('/history')
def history():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    image_folder = os.path.join(current_dir, 'static', 'images')
    suspicious_vehicles = []
    normal_vehicles = []

    try:
        image_files = [f for f in os.listdir(image_folder) if f.startswith('result_') and f.endswith(('.jpg', '.png'))]
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)), reverse=True)

        for img in image_files:
            basename = img.replace('result_', '')
            json_file = os.path.join(image_folder, os.path.splitext(basename)[0] + '.json')
            vehicle = {'image': img, 'data': {}}

            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as jf:
                    vehicle['data'] = json.load(jf)
                if vehicle['data'].get('suspicious'):
                    suspicious_vehicles.append(vehicle)
                else:
                    normal_vehicles.append(vehicle)
            else:
                normal_vehicles.append(vehicle)

    except Exception as e:
        print("历史记录读取失败：", e)

    return render_template('history.html',
                           suspicious_vehicles=suspicious_vehicles,
                           normal_vehicles=normal_vehicles)


@app.route('/prepare')
def warm_up():
    """准备服务，加载所有模型和数据库"""
    global vehicle_info_database
    if load_all_models():
        vehicle_info_database = prepare_service()
        if vehicle_info_database is not None:
            return jsonify({"status": "success", "message": "服务准备完成"})
        else:
            return jsonify({"status": "error", "message": "数据库加载失败"})
    else:
        return jsonify({"status": "error", "message": "模型加载失败"})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    error = None
    if request.method == 'POST':
        # 在这里可以添加真实的用户名和密码验证逻辑
        # 为简化，我们允许任何输入
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            session['logged_in'] = True
            return redirect(url_for('analyze'))
        else:
            error = '请输入用户名和密码'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """退出登录"""
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', methods=['POST', 'GET'])
def analyze():
    """主页面和分析接口"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    global vehicle_info_database
    
    if request.method == 'POST':
        try:
            f = request.files['file']
            if not f or not f.filename:
                return jsonify({"error": 1000, "msg": "未选择文件"})
            
            if not allowed_file(f.filename):
                return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
            
            # 保存上传的文件
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'static/images')
            os.makedirs(upload_dir, exist_ok=True)
            
            filename = secure_filename(f.filename)
            if not filename:
                 return jsonify({"error": 1002, "msg": "无效的文件名"})

            upload_path = os.path.join(upload_dir, filename)
            f.save(upload_path)
            
            # 检查数据库是否已加载
            if vehicle_info_database is None:
                return jsonify({"error": 1003, "msg": "车辆数据库未加载，请先点击'服务准备'"})
            
            # 执行预测
            plate_no, v_type, v_color, car_brand, predict_result, belt_result, result_image = predict(upload_path, vehicle_info_database)

            # 保存带有检测框的图片
            result_filename = f"result_{filename}"
            result_path = os.path.join(upload_dir, result_filename)
            cv2.imwrite(result_path, result_image)
            
            # 保存 JSON 数据（用于历史记录展示）
            json_data = {
                "plate": plate_no,
                "vehicle_type": v_type,
                "color": v_color,
                "brand": car_brand,
                "suspicious": "套牌" in predict_result,  # 判断是否为套牌
                "belt": belt_result
            }
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(upload_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json_data, jf, ensure_ascii=False, indent=2)

            # 构建结果上下文
            context = [
                f"车牌号：{plate_no}",
                f"车型：{v_type}", 
                f"车辆颜色：{v_color}", 
                f"车辆品牌：{car_brand}", 
                f"结论：{predict_result}",
                f"安全带检测：{belt_result}"
            ]
            
            # 将处理后的图片路径（相对于 static 目录）传递给前端
            processed_image_path = os.path.join('images', result_filename).replace('\\', '/')

            return render_template('index.html', context=context, uploaded_image=processed_image_path, val1=time.time())
            
        except Exception as e:
            print(f"分析过程中发生错误: {str(e)}")
            error_context = [
                "车牌号：分析失败",
                "车型：分析失败", 
                "车辆颜色：分析失败", 
                "车辆品牌：分析失败", 
                f"结论：系统错误 - {str(e)}"
            ]
            return render_template('index.html', context=error_context, val1=time.time())
    
    # 首次 GET 访问，暂无结果与图片
    return render_template('index.html', context=None, uploaded_image=None, val1=time.time())

# ===== 颜色与车牌辅助函数 =====

def _pre_for_color(img: np.ndarray) -> np.ndarray:
    """Resize to 64x64 and normalize for color classifier"""
    img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
    return np.expand_dims(img, 0)


def predict_vehicle_color(full_img_bgr: np.ndarray) -> str:
    """两阶段：车辆检测 -> 颜色分类"""
    global yolo_color_model, color_model
    if yolo_color_model is None or color_model is None:
        return "颜色模型未加载"
        
    results = yolo_color_model.predict(full_img_bgr, imgsz=640, verbose=False)
    
    if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
        return "未识别"

    boxes   = results[0].boxes.xyxy.cpu().numpy()
    if boxes.size == 0:
        return "未识别"

    confidences = results[0].boxes.conf
    if confidences is None:
        return "未识别"
        
    x1,y1,x2,y2 = boxes[int(np.argmax(confidences.cpu().numpy()))].astype(int)
    crop = full_img_bgr[y1:y2, x1:x2]
    preds = color_model.predict(_pre_for_color(crop), verbose=0)[0]
    return COLOR_LABELS[int(np.argmax(preds))]

def detect_seat_belt(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    输入图像，返回标注后的图像和文字结论（佩戴/未佩戴/不确定）
    """
    global car_model, belt_model

    # 如果模型未加载，直接返回提示信息，避免 NoneType.predict 错误
    if car_model is None or belt_model is None:
        return image, "安全带模型未加载"

    result_text = "未检测到车辆"

    car_results = car_model.predict(source=image, conf=0.5)
    if not car_results or not car_results[0].boxes:
        return image, result_text

    car_box = car_results[0].boxes.xyxy[0]
    x1, y1, x2, y2 = map(int, car_box)
    roi = image[y1:y2, x1:x2].copy()

    belt_results = belt_model.predict(source=roi, conf=0.3)
    if not belt_results or not belt_results[0].boxes:
        result_text = "未佩戴安全带"
        return image, result_text

    with_cnt = 0
    no_cnt = 0
    uncertain_cnt = 0

    for box, cls, conf in zip(belt_results[0].boxes.xyxy, belt_results[0].boxes.cls, belt_results[0].boxes.conf):
        x1_roi, y1_roi, x2_roi, y2_roi = map(int, box)
        confidence = conf.item()
        if confidence >= 0.5:
            label = 'with_belt' if int(cls) == 1 else 'no_belt'
            if label == 'with_belt':
                with_cnt += 1
                color = (0, 255, 0)
            else:
                no_cnt  += 1
                color = (0, 0, 255)
        else:
            label = 'uncertain'
            uncertain_cnt += 1
            color = (0, 165, 255)

        cv2.rectangle(roi, (x1_roi, y1_roi), (x2_roi, y2_roi), color, 2)
        cv2.putText(roi, f'{label} {confidence:.2f}', (x1_roi, y1_roi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 构建结果文本
    parts = []
    if with_cnt:
        parts.append(f"已佩戴安全带：{with_cnt}")
    if no_cnt:
        parts.append(f"未佩戴安全带：{no_cnt}")
    if uncertain_cnt:
        parts.append(f"安全带状态不确定：{uncertain_cnt}")
    final_label = "，".join(parts) if parts else "安全带状态不确定"

    # 将检测框贴回原图
    image[y1:y2, x1:x2] = roi
    return image, final_label


def detect_plate_number_plate_module(full_img_bgr: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """使用现代YOLOv8接口和plate_recognition模块返回首个车牌号和矩形框"""
    global plate_det_model, plate_rec_model, device
    if plate_det_model is None:
        return "车牌模型未加载", None
    try:
        outputs = plate_det_model.predict(full_img_bgr, conf=0.3, iou=0.5, device=device, verbose=False)

        if not outputs or not hasattr(outputs[0], 'boxes') or outputs[0].boxes is None:
            return "未识别", None
        
        # 只处理置信度最高的车牌
        boxes = outputs[0].boxes
        if len(boxes) == 0:
            return "未识别", None

        confidences = boxes.conf
        if confidences is None:
            return "未识别", None

        best_box_idx = np.argmax(confidences.cpu().numpy())
        box = boxes[best_box_idx]

        rect = box.xyxy[0].cpu().numpy().astype(int)
        label = int(box.cls[0].cpu().numpy())
        
        roi_img = full_img_bgr[rect[1]:rect[3], rect[0]:rect[2]]
        
        if label == 1: # 双层车牌
            roi_img = get_split_merge(roi_img)

        plate_rec_result = get_plate_result(roi_img, device, plate_rec_model, is_color=True)

        if plate_rec_result and len(plate_rec_result) > 0:
            return plate_rec_result[0], rect  # 返回车牌号和矩形框
            
    except Exception as e:
        print(f"plate detect error: {e}")
    return "未识别", None

# ==================== 结果导出与技术支持 ====================

# 导出历史 JSON 记录为 ZIP
@app.route('/export/download', methods=['GET'])
def export_download():
    """将历史 JSON 记录打包为 ZIP 并提供下载"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    image_folder = os.path.join(current_dir, 'static', 'images')
    if not os.path.exists(image_folder):
        return jsonify({"status": "error", "message": "没有历史记录可导出"})

    json_files = [f for f in os.listdir(image_folder) if f.endswith('.json')]

    if not json_files:
        return jsonify({"status": "error", "message": "没有历史记录可导出"})

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for jf in json_files:
            zf.write(os.path.join(image_folder, jf), arcname=jf)

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip',
                     download_name='history_jsons.zip', as_attachment=True)


# 结果导出页面
@app.route('/export')
def export_page():
    """结果导出界面，展示记录数量并提供下载按钮"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    image_folder = os.path.join(current_dir, 'static', 'images')
    if not os.path.exists(image_folder):
        json_count = 0
    else:
        json_files = [f for f in os.listdir(image_folder) if f.endswith('.json')]
        json_count = len(json_files)

    return render_template('export.html', json_count=json_count)

# 技术支持文档页面
@app.route('/support')
def support():
    """技术支持文档页面"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('support.html')

if __name__ == '__main__':
    # 启动时加载模型和数据库
    print("正在加载模型和数据库...")
    if load_all_models():
        vehicle_info_database = prepare_service()
        if vehicle_info_database is not None:
            print("服务启动成功")
        else:
            print("警告: 数据库加载失败")
    else:
        print("警告: 模型加载失败")
    
    app.run(port=8090, debug=True)
