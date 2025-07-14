#x1x4_g版本
import supervision as sv
import mediapipe as mp
import time
import os
import sys
import requests
import numpy as np
import pygame
import cv2
import threading
import math
from datetime import datetime
from ultralytics import YOLO
from linebot import LineBotApi
from linebot.models import ImageSendMessage
## ========== 印出工作目錄 ==========
print("[DEBUG] 當前工作目錄:", os.getcwd())
## ========== 攝影機與畫面 ==========
SOURCE_CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
## ========== YOLO模型 ==========
MODEL = "best.pt"
CROSSING_CLASS = 0
## ========== LINE/Imgur ==========
LINE_BOT_TOKEN = 'kr89Md7IdVNmz98BBnbtgteM5OmKPQEQstDypON1H27R00UZ8BAcKA/HwDToRa+0SoN88g3TYG+PXqujIFjdsLErRFvEmSU9EFuoVdAgyTez0SmsXNWwwXaFgkOp+W6MPf9M/iz8FNrOf6ycikDniAdB04t89/1O/w1cDnyilFU='
LINE_USER_ID = 'U6edec5f059835635e1d487ed36c907cb'
IMGUR_CLIENT_ID = '44338750d0cf7bd'
SCREENSHOT_DIR = r"C:\jt"
MAX_SCREENSHOTS = 10    #最多自動截圖與推播的次數上限
## ========== 建立截圖資料夾 ==========
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
## ========== 亮度控制 ==========
BRIGHTNESS_THRESHOLD = 30       #判斷攝影機畫面是否過暗的亮度下限
BRIGHTNESS_LOW_COUNT_LIMIT = 10 #畫面亮度過低的連續偵測次數上限。
## ========== 手勢magic number ==========
Z_THRESHOLD_RATIO = 0.18    #Z軸深度判斷的比例門檻
XY_THRESHOLD = 0.18         #平面距離（x, y）判斷的比例門檻
FINGER_HOLD_REQUIRED = 1    #食指比出時畫線時間
VICTORY_HOLD_REQUIRED = 1   #剪刀手比出時畫線時間
OPEN_HAND_DIST_RATIO = 0.03 #判斷手掌是否完全張開的常數
## ========== 狀態變數 ==========
left_index_start_time = None    #紀錄左/右食指進入偵測區域的起始時間戳。
right_index_start_time = None   
left_line_locked_x = None       #鎖定左/右手食指進入區域時的X座標。
right_line_locked_x = None
left_line_active = False        #記錄左/右線是否處於「激活」狀態。
right_line_active = False
victory_start_time = None       #紀錄「勝利手勢」開始出現的時間戳。
brightness_low_count = 0        #記錄畫面亮度連續低於門檻的次數。
screenshot_counter = 0          #記錄當前異常狀態下已經自動截圖與推播的次數。
frame_counter = 0               #紀錄目前已處理的影像幀數。
## ========== 環境變數 ==========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['SUPERVISON_DEPRECATION_WARNING'] = '0'
## ========== YOLO模型 ==========
ultralytics = __import__('ultralytics')
ultralytics.checks()
yolo_model = YOLO(MODEL)
yolo_model.fuse()
## ========== MediaPipe ==========
mp_hands = mp.solutions.hands   #引用 MediaPipe 的 Hands 模組
hands = mp_hands.Hands(
    min_detection_confidence=0.7,   #手部檢測的最低信心分數。
    min_tracking_confidence=0.7,    #手部追蹤的最低信心分數。
    max_num_hands=4                 #同時最多偵測 4 隻手。
)
mp_drawing = mp.solutions.drawing_utils #將偵測到的手部關鍵點和連線直接畫在影像上。
## ========== LINE BOT ==========
line_bot_api = LineBotApi(LINE_BOT_TOKEN)
## ========== Supervision工具 ==========
yolo_tracker = sv.ByteTrack()   #建立一個 ByteTrack 追蹤器。
yolo_box_annotator = sv.BoundingBoxAnnotator()  #建立一個 邊界框標註器。
yolo_label_annotator = sv.LabelAnnotator()  #建立一個 標籤標註器。
trace_annotator = sv.TraceAnnotator()   #建立一個 軌跡標註器。
## ========== 計算2D歐氏距離（Euclidean distance）==========
def euclidean_2d(a, b):
    return math.hypot(a.x-b.x, a.y-b.y) #計算食指指尖與拇指指尖的距離
## ========== 計算手掌的「Z軸寬度」==========
def get_hand_z_width(landmarks):
    mp_h = mp.solutions.hands.HandLandmark
    return abs(landmarks[mp_h.THUMB_MCP].z - landmarks[mp_h.PINKY_MCP].z)   #反映手部張開的深度（由掌緣到掌緣）
## ========== 比出食指 ==========
def is_only_index_up_overhead(landmarks):
    mp_h = mp.solutions.hands.HandLandmark
    z_width = get_hand_z_width(landmarks)
    z_threshold = Z_THRESHOLD_RATIO * z_width
    wrist = landmarks[mp_h.WRIST]
    index_tip = landmarks[mp_h.INDEX_FINGER_TIP]
    xy_dist = euclidean_2d(index_tip, wrist)
    index_up = (landmarks[mp_h.INDEX_FINGER_MCP].z - index_tip.z) > z_threshold and xy_dist > XY_THRESHOLD
    others_down = all(
        not ((landmarks[mcp].z - landmarks[tip].z) > z_threshold and euclidean_2d(landmarks[tip], wrist) > XY_THRESHOLD)
        for tip, mcp in [
            (mp_h.MIDDLE_FINGER_TIP, mp_h.MIDDLE_FINGER_MCP),
            (mp_h.RING_FINGER_TIP, mp_h.RING_FINGER_MCP),
            (mp_h.PINKY_TIP, mp_h.PINKY_MCP),
            (mp_h.THUMB_TIP, mp_h.THUMB_MCP)
        ]
    )
    return index_up and others_down
## ========== 比出剪刀手 ==========
def is_only_victory_overhead(landmarks):
    mp_h = mp.solutions.hands.HandLandmark
    z_width = get_hand_z_width(landmarks)
    z_threshold = Z_THRESHOLD_RATIO * z_width
    wrist = landmarks[mp_h.WRIST]
    index_tip = landmarks[mp_h.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_h.MIDDLE_FINGER_TIP]
    index_up = (landmarks[mp_h.INDEX_FINGER_MCP].z - index_tip.z) > z_threshold and euclidean_2d(index_tip, wrist) > XY_THRESHOLD
    middle_up = (landmarks[mp_h.MIDDLE_FINGER_MCP].z - middle_tip.z) > z_threshold and euclidean_2d(middle_tip, wrist) > XY_THRESHOLD
    ring_down = not ((landmarks[mp_h.RING_FINGER_MCP].z - landmarks[mp_h.RING_FINGER_TIP].z) > z_threshold and euclidean_2d(landmarks[mp_h.RING_FINGER_TIP], wrist) > XY_THRESHOLD)
    pinky_down = not ((landmarks[mp_h.PINKY_MCP].z - landmarks[mp_h.PINKY_TIP].z) > z_threshold and euclidean_2d(landmarks[mp_h.PINKY_TIP], wrist) > XY_THRESHOLD)
    thumb_down = not ((landmarks[mp_h.THUMB_MCP].z - landmarks[mp_h.THUMB_TIP].z) > z_threshold and euclidean_2d(landmarks[mp_h.THUMB_TIP], wrist) > XY_THRESHOLD)
    index_middle_dist = euclidean_2d(index_tip, middle_tip)
    ring_tip = landmarks[mp_h.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_h.PINKY_TIP]
    min_other_dist = min(
        euclidean_2d(index_tip, ring_tip),
        euclidean_2d(middle_tip, ring_tip),
        euclidean_2d(ring_tip, pinky_tip),
        euclidean_2d(middle_tip, pinky_tip)
    )
    spread_ok = index_middle_dist > min_other_dist * 1.5
    return index_up and middle_up and ring_down and pinky_down and thumb_down and spread_ok
## ========== 調整影像的色相（Hue）與亮度（Brightness） ==========
#   hue_shift：色相偏移量，單位為 HSV 色彩空間的 degree（範圍 0~179），預設為 60。
#   brightness_scale：亮度縮放倍率，預設為 1.0（不變）。
def adjust_hue_and_brightness(frame, hue_shift=60, brightness_scale=1.0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_scale, 0, 230)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
## ========== 播放音效 ==========
def play_start_sound():
    def _play():
        file_path = r'start.mp3'
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            time.sleep(3)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"[play_start_sound異常] {e}")
    threading.Thread(target=_play, daemon=True).start()
##  ========== LINE/Imgur/推播物件 ==========
class LineNotifier:
    def __init__(self, token, user_id):
        self.token = token
        self.user_id = user_id
##  上傳圖片到 Imgur
    def upload_to_imgur(self, image_path):
        try:
            with open(image_path, "rb") as f:
                files = {'image': f}
                headers = {'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'}
                response = requests.post(
                    "https://api.imgur.com/3/image", files=files, headers=headers, timeout=10
                )
                response.raise_for_status()
                link = response.json()['data']['link']
                return link
        except Exception as e:
            print(f"[Imgur上傳失敗] {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"詳細錯誤: {e.response.status_code} / {e.response.text}")
            return None
##  文字廣播方法    
    def broadcast_message(self, Tmessage):
        print(f"輸入的訊息是: {Tmessage}")
        url = "https://api.line.me/v2/bot/message/broadcast"
        headers = {
            "Authorization": f"Bearer {LINE_BOT_TOKEN}",
            "Content-Type": "application/json"
        }
        body = {
            "messages":[
                {
                    "type": "text",
                    "text": Tmessage
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers , json=body)
            response.raise_for_status()
            print(f"消息廣播成功 回應: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"無法廣播: {e}")
            if e.response:
                print(f"回應內容: {e.response.text}")
##  私訊圖片給特定 LINE 用戶
    def send_image_to_line(self, imgur_link):
        try:
            line_bot_api.push_message(
                self.user_id,
                ImageSendMessage(
                    original_content_url=imgur_link,
                    preview_image_url=imgur_link
                )
            )
            return True
        except Exception as e:
            print(f"[LINE推播失敗] {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"詳細錯誤: {e.response.status_code} / {e.response.text}")
            return False
##  廣播圖片到全體 LINE 用戶
    def broadcast_img(self, imgur_link):
        url = "https://api.line.me/v2/bot/message/broadcast"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        body = {
            "messages":[
                {
                    "type": "image",
                    "originalContentUrl": imgur_link,
                    "previewImageUrl": imgur_link
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"[LINE Broadcast失敗] {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"詳細錯誤: {e.response.status_code} / {e.response.text}")
            return False
## 異步推播截圖（或圖片）
    def async_screenshot_notify(self, img_path):
        def task():
            try:
                imgur_link = self.upload_to_imgur(img_path)
                if imgur_link:
                    self.send_image_to_line(imgur_link)
                    self.broadcast_img(imgur_link)
                    self.broadcast_message("檢測到錯誤")
                else:
                    print("[推播失敗] Imgur未取得連結")
            except Exception as e:
                print(f"[推播異常] {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"詳細錯誤: {e.response.status_code} / {e.response.text}")
        threading.Thread(target=task, daemon=True).start()
#   實例化物件
notifier = LineNotifier(LINE_BOT_TOKEN, LINE_USER_ID)
## ========== YOLO判別 =========    
def run_yolo_on_frame(frame):
    #1. 用YOLO模型對輸入frame做物件偵測，取得結果
    results = yolo_model(frame, verbose=False)[0]    
    #2. 將YOLO的結果轉為 supervision 套件的 Detections 物件格式（未 tracking）
    detections = sv.Detections.from_ultralytics(results)    
    #3. 回傳原始 frame 及 detections（未 tracking）
    return frame, detections
## ========== 亮度檢測 ==========
def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)
## ========== 重啟程式 =========
def restart_program():
    python = sys.executable
    os.execv(python, [python] + sys.argv)
## ========== 主程式 ==========
def main():
    #   全域變數重設
    global left_index_start_time, right_index_start_time
    global left_line_locked_x, right_line_locked_x
    global left_line_active, right_line_active
    global victory_start_time, brightness_low_count, screenshot_counter, frame_counter
    #   動態篩選狀態
    after_3600_frames = False
    error_timer = 0  # 用於 class 6 連續偵測計數
    #   攝影機初始化
    cap = cv2.VideoCapture(SOURCE_CAMERA_INDEX, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    #   LineZone Annotator 初始化，物件初始化為 None
    annotator_left = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    annotator_right = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
    linezone_left = None
    linezone_right = None
## ========== 實驗前：偵測雙手手掌張開 ==========
    #   手勢啟動區段的狀態變數
    start_time = None
    ok_duration = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        #   調整影像色相與亮度，提升手部辨識效果。
        processed_frame = adjust_hue_and_brightness(frame, hue_shift=60, brightness_scale=1.0)
        #   轉成 RGB 給 MediaPipe 處理。
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        imgHeight, imgWidth, _ = frame.shape
        results = hands.process(image_rgb)
        open_hands = 0
        #   取得手部關鍵點後，計算五指間距，判斷是否「雙手張開」。
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_h = mp.solutions.hands.HandLandmark
                tips = [hand_landmarks.landmark[f] for f in [
                    mp_h.THUMB_TIP, mp_h.INDEX_FINGER_TIP, mp_h.MIDDLE_FINGER_TIP, mp_h.RING_FINGER_TIP, mp_h.PINKY_TIP
                ]]
                threshold = OPEN_HAND_DIST_RATIO * imgWidth
                dists = [np.linalg.norm(np.array([tips[i].x, tips[i].y]) - np.array([tips[j].x, tips[j].y]))*imgWidth
                         for i in range(5) for j in range(i+1,5)]
                if all([d > threshold for d in dists]):
                    open_hands += 1
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        #   若雙手張開，持續計時到達指定秒數才 break（進入下一階段）。
        if open_hands >= 2:
            #   如果 start_time 是 None，開始計時
            if start_time is None:
                start_time = time.time()
                ok_duration = 0
            #   否則持續累加已張開的秒數，顯示在畫面上 
            else:
                ok_duration = time.time() - start_time
            cv2.putText(frame, "OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Duration: {ok_duration:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #   如果累積張開秒數 ok_duration >= 5秒，就呼叫 play_start_sound() 並 break 跳出 while 迴圈。
            if ok_duration >= 5:
                play_start_sound()
                break
        #   如果沒偵測到雙手張開，則 start_time 歸零、ok_duration 歸零，畫面顯示「Hand Closed」。
        else:
            start_time = None
            ok_duration = 0
            cv2.putText(frame, "Hand Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Hand Detection', frame)
        #   按ESC結束程式
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
## ========== 實驗中流程 ==========
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        #   用於統計處理過的 frame 數，後續判斷異常（如 3600 frame 後的特別處理）。
        frame_counter += 1
        #   取 frame 的亮度值（通常抓灰階均值）。
        brightness = get_brightness(frame)
        #   若亮度低於門檻（BRIGHTNESS_THRESHOLD）：
        if brightness < BRIGHTNESS_THRESHOLD:
            brightness_low_count += 1
            #   若連續低亮 frame 達門檻（BRIGHTNESS_LOW_COUNT_LIMIT）:
            if brightness_low_count >= BRIGHTNESS_LOW_COUNT_LIMIT:
            # 則自動釋放攝影機、關視窗、重啟程式。                
                cap.release()
                cv2.destroyAllWindows()
                restart_program()
        else:
            #   若亮度正常，則 brightness_low_count = 0，避免誤判。
            brightness_low_count = 0
        #   影像預處理與手勢判斷
        processed_frame = adjust_hue_and_brightness(frame, hue_shift=60, brightness_scale=1.0)
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        imgHeight, imgWidth, _ = frame.shape
        results = hands.process(image_rgb)
        #   剪刀手（Victory）判斷
        any_victory = False
        for hand_landmarks in results.multi_hand_landmarks if results.multi_hand_landmarks else []:
            #   用 is_only_victory_overhead 判斷是否有任一手呈現剪刀手。
            if is_only_victory_overhead(hand_landmarks.landmark):
                #   若 victory_start_time 為 None:
                if victory_start_time is None:
                    #   設為現在時間（開始計時）
                    victory_start_time = time.time()
                #   否則計算已維持秒數（held_time）。    
                held_time = time.time() - victory_start_time
                #   持續顯示「Victory: X.XXs」在畫面上。
                cv2.putText(frame, f"剪刀手: {held_time:.2f}s", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                #   若 held_time 達門檻（VICTORY_HOLD_REQUIRED），將 any_victory = True。
                if held_time >= VICTORY_HOLD_REQUIRED:
                    any_victory = True
            else:
                #   若沒有 victory，且所有手都沒 victory:
                if not any(is_only_victory_overhead(h.landmark) for h in results.multi_hand_landmarks):
                    #   victory_start_time = None（計時歸零）
                    victory_start_time = None
        #   若 any_victory 為 True，且任一警戒線 active，則：
        if any_victory:
            if left_line_active or right_line_active:
                play_start_sound()
                #   重設所有警戒線與相關狀態（進行「線條消除」動作）
                left_line_active = False
                right_line_active = False
                left_line_locked_x = None
                right_line_locked_x = None
                left_index_start_time = None
                right_index_start_time = None
                victory_start_time = None
                screenshot_counter = 0
                linezone_left = None
                linezone_right = None
                cv2.putText(frame, "比出剪刀手，線條已消除", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)
        #   Index Up（食指舉起）判斷與警戒線建立
        if results.multi_hand_landmarks and results.multi_handedness:
            #   對每隻手分開判斷（根據 handedness）
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                landmarks = hand_landmarks.landmark
                base_y = 60 if label == 'Left' else 200
                #   用 is_only_index_up_overhead 判斷是否只有食指舉起。
                index_up = is_only_index_up_overhead(landmarks)
                tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
                diff_z = mcp.z - tip.z
                #   取出 INDEX_FINGER_TIP 與 MCP 的 z 座標，顯示於畫面（有助於 debug 手部深度）。
                cv2.putText(frame, f"{label} TIP z:{tip.z:.3f}", (30, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.putText(frame, f"{label} MCP z:{mcp.z:.3f}", (30, base_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"{label} diff z:{diff_z:.3f}", (30, base_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                #   若 index_up 為 True：
                if index_up:
                    #   顯示「Index UP!」
                    cv2.putText(frame, "Index UP!", (30, base_y+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                else:
                    cv2.putText(frame, "Index DOWN", (30, base_y+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,128,128), 3)
                if label == 'Left':
                    #   若對應警戒線尚未 active，且 start_time 為 None:開始計時。
                    if not left_line_active and index_up:
                        if left_index_start_time is None:
                            left_index_start_time = time.time()
                        held_time = time.time() - left_index_start_time
                        #   若持續 index_up 達 FINGER_HOLD_REQUIRED 秒，且尚未鎖定 x:
                        if held_time >= FINGER_HOLD_REQUIRED and left_line_locked_x is None:
                            #   鎖定該食指 x 座標為警戒線位置
                            left_line_locked_x = int(landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * imgWidth)
                            #   設定該側警戒線 active
                            left_line_active = True
                            play_start_sound()
                    #   若食指未舉起       
                    elif not index_up:
                        #   則 start_time 歸零。
                        left_index_start_time = None
                #右手判別        
                else:
                    if not right_line_active:
                        if index_up:
                            if right_index_start_time is None:
                                right_index_start_time = time.time()
                            held_time = time.time() - right_index_start_time
                            cv2.putText(frame, f"Right Index: {held_time:.2f}s", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                            if held_time >= FINGER_HOLD_REQUIRED and right_line_locked_x is None:
                                right_line_locked_x = int(landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * imgWidth)
                                right_line_active = True
                                play_start_sound()
                        else:
                            right_index_start_time = None
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        #   若左/右警戒線 active 且 x 座標已鎖定，則建立對應的 LineZone 物件。
        if left_line_active and left_line_locked_x is not None:
            if linezone_left is None:
                linezone_left = sv.LineZone(
                    start=sv.Point(left_line_locked_x, 0),
                    end=sv.Point(left_line_locked_x, imgHeight)
                )
        if right_line_active and right_line_locked_x is not None:
            if linezone_right is None:
                linezone_right = sv.LineZone(
                    start=sv.Point(right_line_locked_x, 0),
                    end=sv.Point(right_line_locked_x, imgHeight)
                )
        try:
            #   對 frame 執行 YOLO 偵測，取得所有目標物件與追蹤結果。
            frame, detections = run_yolo_on_frame(frame)
            #若有 class 6（假設為異常類型），error_timer 累加
            if 6 in detections.class_id:
                error_timer += 1
            else:
                #   否則歸零
                error_timer = 0
            #   若 error_timer 達 900:    
            if error_timer >= 900:
                #   設定 after_3600_frames 為 True（進入異常狀態）
                after_3600_frames = True
            #   根據 after_3600_frames 狀態，過濾需監控的 class   
            if after_3600_frames:
                filtered = detections[np.isin(detections.class_id, [0, 2, 4, 5])]
            else:
                filtered = detections[~np.isin(detections.class_id, [1])]           
            #   更新追蹤（yolo_tracker）。
            filtered = yolo_tracker.update_with_detections(filtered)
            #   產生標籤
            labels = [
                f"#{tracker_id} {yolo_model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(filtered.confidence, filtered.class_id, filtered.tracker_id)
            ]
            #   畫出 YOLO 偵測框、標註、追蹤路徑。
            frame = yolo_box_annotator.annotate(scene=frame, detections=filtered)
            frame = yolo_label_annotator.annotate(scene=frame, detections=filtered, labels=labels)
            frame = trace_annotator.annotate(scene=frame, detections=filtered)
            #   若 frame_counter > 3600 且有偵測到物件
            if frame_counter > 3600 and len(filtered) > 0:
                #   顯示錯誤，並執行截圖與推播（最多 MAX_SCREENSHOTS 次）
                cv2.putText(frame, "Error (after 3600 frames)", (9, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                if screenshot_counter < MAX_SCREENSHOTS:
                    now = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path = os.path.join(SCREENSHOT_DIR, f"error_screenshot_{now}.png")
                    success = cv2.imwrite(img_path, frame)
                    if success:
                        notifier.async_screenshot_notify(img_path)
                        screenshot_counter += 1
                    else:
                        cv2.putText(frame, "推播失敗", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
            #   若 linezone_left/right 有 crossing（in_count > 0）：
            if linezone_left is not None:
                linezone_left.trigger(filtered)
                frame = annotator_left.annotate(frame, line_counter=linezone_left)
                cv2.putText(frame, "Left Index Line", (left_line_locked_x + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
                if linezone_left.in_count > 0:
                    #   顯示 Error，執行截圖與推播
                    cv2.putText(frame, "Error", (9, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                    if screenshot_counter < MAX_SCREENSHOTS:
                        now = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_path = os.path.join(SCREENSHOT_DIR, f"error_screenshot_left_{now}.png")
                        success = cv2.imwrite(img_path, frame)
                        if success:
                            notifier.async_screenshot_notify(img_path)
                            screenshot_counter += 1
                        else:
                            cv2.putText(frame, "推播失敗", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    #   in_count 歸零。
                    linezone_left.in_count = 0
                cv2.putText(frame, f"in: {linezone_left.in_count}", (left_line_locked_x + 10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
                cv2.putText(frame, f"out: {linezone_left.out_count}", (left_line_locked_x + 10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)            
            #   若有物件 crossing 右警戒線，顯示 Error，並進行截圖與推播，in_count 歸零。
            if linezone_right is not None:
                linezone_right.trigger(filtered)
                frame = annotator_right.annotate(frame, line_counter=linezone_right)
                cv2.putText(frame, "Right Index Line", (right_line_locked_x + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
                if linezone_right.in_count > 0:
                    cv2.putText(frame, "Error", (9, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                    if screenshot_counter < MAX_SCREENSHOTS:
                        now = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_path = os.path.join(SCREENSHOT_DIR, f"error_screenshot_right_{now}.png")
                        success = cv2.imwrite(img_path, frame)
                        if success:
                            notifier.async_screenshot_notify(img_path)
                            screenshot_counter += 1
                        else:
                            cv2.putText(frame, "推播失敗", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    linezone_right.in_count = 0
                cv2.putText(frame, f"in: {linezone_right.in_count}", (right_line_locked_x + 10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
                cv2.putText(frame, f"out: {linezone_right.out_count}", (right_line_locked_x + 10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
        
        except Exception as e:  #捕捉並回報處理 frame（如前面影像處理、手勢分析等）時發生的任何例外錯誤。
            print("處理frame時發生錯誤：", e)
        try:    #捕捉並回報處理 frame（如前面影像處理、手勢分析等）時發生的任何例外錯誤。
            cv2.imshow("Mediapipe Hue+Brightness", processed_frame)
            cv2.imshow('Index LineZone Persistent', frame)
        except Exception as e:  #如果 cv2.imshow 過程發生任何錯誤（如視窗無法開啟、frame 資料型態錯誤等），會捕捉並輸出錯誤訊息 [cv2.imshow異常] {e}。
            print(f"[cv2.imshow異常] {e}")
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
## ========== Python主程式的標準啟動入口。 ==========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("主程式發生錯誤：", e)