import cv2
import os
import time
import datetime
import glob
import requests
import json
import numpy as np
import threading
from collections import deque

# COCO Wholebody 스켈레톤 연결 정보
COCO_WHOLEBODY_SKELETON = [
    # Body (0~16)
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],

    # Left Hand (91~111)
    [91, 92], [92, 93], [93, 94], [94, 95],         # Thumb
    [91, 96], [96, 97], [97, 98], [98, 99],         # Index
    [91, 100], [100, 101], [101, 102], [102, 103],  # Middle
    [91, 104], [104, 105], [105, 106], [106, 107],  # Ring
    [91, 108], [108, 109], [109, 110], [110, 111],  # Pinky

    # Right Hand (112~132)
    [112, 113], [113, 114], [114, 115], [115, 116],      # Thumb
    [112, 117], [117, 118], [118, 119], [119, 120],      # Index
    [112, 121], [121, 122], [122, 123], [123, 124],      # Middle
    [112, 125], [125, 126], [126, 127], [127, 128],      # Ring
    [112, 129], [129, 130], [130, 131], [131, 132],      # Pinky
]

def draw_keypoints_wholebody_on_frame(frame, keypoints, scores, threshold=2.0):
    """프레임에 wholebody 키포인트와 스켈레톤을 그리는 함수 (서버에서 이미 좌표 변환됨)"""
    num_points = 133
    
    print(f"🔧 스켈레톤 그리기 시작")
    print(f"  - 프레임 크기: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  - 키포인트 개수: {len(keypoints)}")
    print(f"  - 스코어 개수: {len(scores)}")

    # 키포인트 그리기 (서버에서 이미 원본 이미지 좌표로 변환됨)
    drawn_points = 0
    for idx in range(min(num_points, len(keypoints), len(scores))):
        if 17 <= idx <= 22:  # 발 keypoint 무시
            continue
        if scores[idx] > threshold:
            x, y = keypoints[idx][:2]
            # 프레임 경계 확인
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                drawn_points += 1

    print(f"✅ 그려진 키포인트: {drawn_points}개")

    # 스켈레톤 연결선 그리기
    drawn_lines = 0
    for idx1, idx2 in COCO_WHOLEBODY_SKELETON:
        if 17 <= idx1 <= 22 or 17 <= idx2 <= 22:  # 발 keypoint 포함된 연결 무시
            continue
        if (idx1 < len(scores) and idx2 < len(scores) and 
            scores[idx1] > threshold and scores[idx2] > threshold):
            x1, y1 = keypoints[idx1][:2]
            x2, y2 = keypoints[idx2][:2]
            
            # 프레임 경계 확인
            if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                drawn_lines += 1
    
    print(f"✅ 그려진 연결선: {drawn_lines}개")

def draw_sign_prediction_on_frame(frame, prediction_text, confidence, x=10, y=30):
    """프레임에 수어 인식 결과를 그리는 함수"""
    if prediction_text:
        # 배경 박스 그리기
        text = f"Sign: {prediction_text} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 배경 사각형
        cv2.rectangle(frame, (x-5, y-text_height-10), (x+text_width+5, y+baseline+5), (0, 0, 0), -1)
        
        # 텍스트
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

class WebcamCapture:
    """웹캠 캡처 및 실시간 처리 클라이언트"""
    
    def __init__(self, server_url="http://000.000.000.000:5000"):
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.capturing_images = False
        self.realtime_translate = False
        self.realtime_fps = 0
        self.capture_folder = None
        self.capture_image_count = 0
        self.video_folder = "./captured_videos"
        self.image_folder = "./captured_images"
        self.video_count = 0
        self.image_count = 0
        self.w = 640
        self.h = 480
        self.fps = 10
        self.last_server_result = ""
        self.server_url = server_url
        
        # 스켈레톤 시각화 관련 변수들
        self.show_skeleton = False
        self.last_pose_data = None  # 마지막 포즈 데이터 저장
        self.pose_threshold = 2.0

        # 수어 인식 관련 변수들
        self.sign_recognition_mode = False
        self.show_sign_prediction = False
        self.last_sign_prediction = ""
        self.last_sign_confidence = 0.0
        self.sign_confidence_threshold = 0.6
        self.client_id = f"webcam_{int(time.time())}"
        
        # 수어 인식 예측 결과 평활화를 위한 버퍼
        self.sign_prediction_buffer = deque(maxlen=5)
        self.sign_smoothing_enabled = True

        # 폴더 생성
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)

    def initialize_camera(self):
        """카메라 초기화"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        return self.cap.isOpened()
    
    def release_camera(self):
        """카메라 해제"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def toggle_skeleton_display(self):
        """스켈레톤 표시 토글"""
        self.show_skeleton = not self.show_skeleton
        status = f"스켈레톤 표시: {'켜짐' if self.show_skeleton else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def toggle_sign_recognition(self):
        """수어 인식 모드 토글 (수정됨 - 포즈 추정과 동시 실행 가능)"""
        self.sign_recognition_mode = not self.sign_recognition_mode
        if self.sign_recognition_mode:
            # 기존의 포즈 추정 강제 종료 코드 제거
            # self.realtime_translate = False  # 이 줄을 주석 처리하거나 제거
            self.clear_sign_buffer()
        status = f"수어 인식 모드: {'켜짐' if self.sign_recognition_mode else '꺼짐'}"
        print(f"🤟 {status}")
        return status
    
    def toggle_sign_prediction_display(self):
        """수어 예측 결과 표시 토글"""
        self.show_sign_prediction = not self.show_sign_prediction
        status = f"수어 예측 표시: {'켜짐' if self.show_sign_prediction else '꺼짐'}"
        print(f"🎯 {status}")
        return status
    
    def toggle_sign_smoothing(self):
        """수어 예측 평활화 토글"""
        self.sign_smoothing_enabled = not self.sign_smoothing_enabled
        if not self.sign_smoothing_enabled:
            self.sign_prediction_buffer.clear()
        status = f"수어 예측 평활화: {'켜짐' if self.sign_smoothing_enabled else '꺼짐'}"
        print(f"📊 {status}")
        return status
    
    def set_pose_threshold(self, threshold):
        """포즈 임계값 설정"""
        self.pose_threshold = threshold
        return f"포즈 임계값: {threshold}"
    
    def set_sign_confidence_threshold(self, threshold):
        """수어 인식 신뢰도 임계값 설정"""
        self.sign_confidence_threshold = threshold
        return f"수어 신뢰도 임계값: {threshold}"
    
    def clear_sign_buffer(self):
        """서버의 수어 특징 버퍼 클리어"""
        try:
            response = requests.post(f"{self.server_url}/clear_buffer/{self.client_id}", timeout=5)
            if response.status_code == 200:
                print(f"✅ 수어 버퍼 클리어됨: {self.client_id}")
            self.sign_prediction_buffer.clear()
            self.last_sign_prediction = ""
            self.last_sign_confidence = 0.0
        except Exception as e:
            print(f"❌ 수어 버퍼 클리어 실패: {e}")
    
    def start_realtime(self):
        """이미지 실시간 번역 시작 (기존 포즈 추정)"""
        if not self.realtime_translate:
            self.realtime_translate = True
            self.realtime_fps = 0
            self.last_server_result = ""
            # 수어 인식 모드 비활성화
            # self.sign_recognition_mode = False
            return f"실시간 포즈 추정 시작"
        return "이미 실시간 포즈 추정 중입니다"

    def stop_realtime(self):
        """이미지 실시간 번역 종료"""
        if self.realtime_translate:
            self.realtime_translate = False
            self.realtime_fps = 0
            return f"실시간 포즈 추정 종료"
        return "이미 실시간 포즈 추정 중이 아닙니다."

    def start_capture_images(self):
        """이미지 연속 저장 시작"""
        if not self.capturing_images:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.capture_folder = os.path.join("captured_datas", now)
            os.makedirs(self.capture_folder, exist_ok=True)
            self.capturing_images = True
            self.capture_image_count = 0
            return f"이미지 캡처 시작: {self.capture_folder}"
        return "이미 이미지 캡처 중입니다"

    def stop_capture_images(self):
        """이미지 연속 저장 종료"""
        if self.capturing_images:
            self.capturing_images = False
            folder = self.capture_folder
            self.capture_folder = None
            return f"이미지 캡처 종료: {folder}"
        return "이미지 캡처 중이 아닙니다"

    def capture_frame(self):
        """프레임 캡처"""
        if not self.initialize_camera():
            return None

        ret, frame = self.cap.read()
        if ret:
            # 실시간 모드에 따른 서버 요청
            if self.realtime_translate:
                self.send_frame_for_pose_estimation(frame)
            if self.sign_recognition_mode:
                self.send_frame_for_sign_recognition(frame)
            
            # 현재 상태 출력 (매 30프레임마다)
            if self.realtime_fps % 30 == 0:
                print(f"📊 현재 상태 - 포즈추정: {self.realtime_translate}, 수어인식: {self.sign_recognition_mode}, "
                      f"스켈레톤표시: {self.show_skeleton}, 수어표시: {self.show_sign_prediction}")
            
            # 스켈레톤 표시가 활성화되어 있고 포즈 데이터가 있으면 그리기
            if self.show_skeleton and self.last_pose_data:
                try:
                    keypoints = self.last_pose_data.get('keypoints', [])
                    scores = self.last_pose_data.get('scores', [])
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, self.pose_threshold
                        )
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
            
            # 수어 예측 결과 표시
            if self.show_sign_prediction and self.last_sign_prediction:
                try:
                    draw_sign_prediction_on_frame(
                        frame, self.last_sign_prediction, self.last_sign_confidence
                    )
                except Exception as e:
                    print(f"수어 예측 표시 오류: {e}")
            
            # 녹화 중이면 비디오에 저장
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            # 이미지 연속 저장 중이면 파일로 저장
            elif self.capturing_images and self.capture_folder:
                self.capture_image_count += 1
                image_path = os.path.join(
                    self.capture_folder, f"img_{self.capture_image_count:04d}.jpg"
                )
                cv2.imwrite(image_path, frame)

            return frame
        return None
    
    def save_image(self, save_skeleton=False, save_sign=False):
        """이미지 저장"""
        if not self.initialize_camera():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # 스켈레톤 저장 옵션이 활성화되어 있고 포즈 데이터가 있으면 그리기
            if save_skeleton and self.last_pose_data:
                try:
                    keypoints = self.last_pose_data.get('keypoints', [])
                    scores = self.last_pose_data.get('scores', [])
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, self.pose_threshold
                        )
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
            
            # 수어 예측 저장 옵션
            if save_sign and self.last_sign_prediction:
                try:
                    draw_sign_prediction_on_frame(
                        frame, self.last_sign_prediction, self.last_sign_confidence
                    )
                except Exception as e:
                    print(f"수어 예측 그리기 오류: {e}")
    
    def start_recording(self):
        """비디오 녹화 시작"""
        if not self.recording:
            self.video_count += 1
            video_path = f"{self.video_folder}/record_{self.video_count}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.w, self.h))
            self.recording = True
            return f"녹화 시작: {video_path}"
        return "이미 녹화 중입니다"
    
    def stop_recording(self):
        """비디오 녹화 종료"""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            return "녹화 종료"
        return "녹화 중이 아닙니다"
    
    def generate_frames(self):
        """스트리밍용 프레임 생성"""
        # fps가 0이거나 None이면 기본값 30fps 사용
        fps = self.fps if self.fps and self.fps > 0 else 30
        target_interval = 1.0 / fps

        while True:
            start = time.time()
            frame = self.capture_frame()
            if frame is not None:
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            elapsed = time.time() - start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_frame_for_pose_estimation(self, frame):
        """서버로 원본 프레임 전송 (포즈 추정용)"""
        self.realtime_fps += 1 
        
        # 원본 이미지를 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ 이미지 인코딩 실패")
            return
            
        frame_bytes = buffer.tobytes()
        
        try:
            print(f"📤 서버로 포즈 추정 이미지 전송")
            
            files = {'image': (f'{self.realtime_fps}.jpg', frame_bytes, 'image/jpeg')}
            data = {'frame_id': str(self.realtime_fps)}
            
            resp = requests.post(
                f"{self.server_url}/estimate_pose",
                files=files,
                data=data,
                timeout=10
            )
            
            # 서버 응답 처리 및 포즈 데이터 저장
            if resp.status_code == 200:
                print("✅ 포즈 추정 서버 전송 성공")
                try:
                    response_data = resp.json()
                    
                    if 'error' in response_data:
                        print(f"⚠️ 서버 오류: {response_data['error']}")
                        return
                    
                    keypoints = response_data.get('keypoints', [])
                    scores = response_data.get('scores', [])
                    person_box = response_data.get('person_box', [])
                    
                    if keypoints and scores:
                        self.last_pose_data = {
                            'keypoints': keypoints,
                            'scores': scores,
                            'person_box': person_box
                        }
                        print(f"✅ 포즈 데이터 업데이트됨!")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 오류: {e}")
                except Exception as e:
                    print(f"❌ 서버 응답 파싱 오류: {e}")
            else:
                print(f"❌ 포즈 추정 서버 전송 실패: {resp.status_code}")
                
            self.last_server_result = f"{self.realtime_fps}-포즈추정결과"
            
        except Exception as e:
            print(f"❌ 포즈 추정 서버 전송 실패: {e}")

    def send_frame_for_sign_recognition(self, frame):
        """서버로 원본 프레임 전송 (수어 인식용)"""
        self.realtime_fps += 1
        
        # 원본 이미지를 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ 이미지 인코딩 실패")
            return
            
        frame_bytes = buffer.tobytes()
        
        try:
            print(f"🤟 서버로 수어 인식 이미지 전송")
            
            files = {'image': (f'{self.realtime_fps}.jpg', frame_bytes, 'image/jpeg')}
            data = {
                'frame_id': str(self.realtime_fps),
                'client_id': self.client_id,
                'confidence_threshold': str(self.sign_confidence_threshold),
                'extract_only': 'false'
            }
            
            resp = requests.post(
                f"{self.server_url}/sign_recognition",
                files=files,
                data=data,
                timeout=10
            )
            
            # 서버 응답 처리
            if resp.status_code == 200:
                print("✅ 수어 인식 서버 전송 성공")
                try:
                    response_data = resp.json()
                    
                    if 'error' in response_data:
                        print(f"⚠️ 서버 오류: {response_data['error']}")
                        return
                    
                    # 수어 예측 결과 처리
                    sign_predictions = response_data.get('sign_predictions', [])
                    buffer_length = response_data.get('buffer_length', 0)
                    sequence_required = response_data.get('sequence_length_required', 32)
                    
                    print(f"📊 수어 버퍼 상태: {buffer_length}/{sequence_required}")
                    
                    if sign_predictions:
                        best_prediction = sign_predictions[0]
                        pred_word = best_prediction['word']
                        pred_confidence = best_prediction['confidence']
                        
                        print(f"🎯 수어 예측: {pred_word} (신뢰도: {pred_confidence:.2f})")
                        
                        # 평활화 처리
                        if self.sign_smoothing_enabled:
                            self.sign_prediction_buffer.append({
                                'word': pred_word,
                                'confidence': pred_confidence
                            })
                            self.smooth_sign_predictions()
                        else:
                            # 평활화 없이 직접 업데이트
                            if pred_confidence >= self.sign_confidence_threshold:
                                self.last_sign_prediction = pred_word
                                self.last_sign_confidence = pred_confidence
                    else:
                        print("⏳ 수어 예측 결과 없음")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 오류: {e}")
                except Exception as e:
                    print(f"❌ 서버 응답 파싱 오류: {e}")
            else:
                print(f"❌ 수어 인식 서버 전송 실패: {resp.status_code}")
                
        except Exception as e:
            print(f"❌ 수어 인식 서버 전송 실패: {e}")

    def smooth_sign_predictions(self):
        """수어 예측 결과 평활화"""
        if not self.sign_prediction_buffer:
            return
        
        # 단어별 신뢰도 점수 집계
        word_scores = {}
        for pred in self.sign_prediction_buffer:
            word = pred['word']
            confidence = pred['confidence']
            
            if word in word_scores:
                word_scores[word].append(confidence)
            else:
                word_scores[word] = [confidence]
        
        # 각 단어의 평균 신뢰도 계산
        word_avg_scores = {}
        for word, scores in word_scores.items():
            word_avg_scores[word] = np.mean(scores)
        
        if word_avg_scores:
            # 가장 높은 평균 신뢰도를 가진 단어 선택
            best_word, best_confidence = max(word_avg_scores.items(), key=lambda x: x[1])
            
            # 임계값 이상이고 기존 예측과 다르면 업데이트
            if best_confidence >= self.sign_confidence_threshold:
                if self.last_sign_prediction != best_word:
                    print(f"🎯 평활화된 수어 예측: {best_word} (신뢰도: {best_confidence:.2f})")
                    self.last_sign_prediction = best_word
                    self.last_sign_confidence = best_confidence

    def process_latest_folder_images(self):
        """가장 최근 폴더의 이미지를 서버로 전송 (포즈 추정용)"""
        base_dir = "captured_datas"
        if not os.path.exists(base_dir):
            return "저장된 폴더가 없습니다."
        
        folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            return "저장된 폴더가 없습니다."
        
        latest_folder = max(folders, key=os.path.getmtime)
        images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")))
        send_count = 0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # 이미지를 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"❌ 이미지 인코딩 실패: {img_path}")
                continue
            
            image_bytes = buffer.tobytes()
            
            # 서버로 전송
            try:
                files = {'image': (os.path.basename(img_path), image_bytes, 'image/jpeg')}
                data = {'frame_id': os.path.basename(img_path)}
                
                resp = requests.post(
                    f"{self.server_url}/estimate_pose",
                    files=files,
                    data=data,
                    timeout=10
                )
                
                if resp.status_code == 200:
                    send_count += 1
                    print(f"✅ 서버로 전송 성공: {send_count}, {img_path}")
                else:
                    print(f"❌ 서버 응답 오류: {resp.status_code} {resp.text}")
                    
            except Exception as e:
                print(f"❌ 서버 전송 실패: {os.path.basename(img_path)} - {e}")
        
        return f"이미지 처리 완료: {len(images)}개 이미지 중 {send_count}개 서버 전송 성공"

    def get_server_stats(self):
        """서버 통계 정보 조회"""
        try:
            resp = requests.get(f"{self.server_url}/stats", timeout=5)
            if resp.status_code == 200:
                stats = resp.json()
                print("\n📊 서버 통계:")
                print(f"  - 포즈 추정 요청: {stats.get('request_count', 0)}")
                print(f"  - 수어 인식 요청: {stats.get('sign_request_count', 0)}")
                print(f"  - 수어 모델 로드됨: {stats.get('sign_model_loaded', False)}")
                print(f"  - 활성 클라이언트: {stats.get('active_clients', 0)}")
                print(f"  - 평균 처리시간: {stats.get('processing_times', {}).get('mean', 0)*1000:.1f}ms")
                print(f"  - 평균 수어 처리시간: {stats.get('sign_processing_times', {}).get('mean', 0)*1000:.1f}ms")
                return stats
            else:
                print(f"❌ 서버 통계 조회 실패: {resp.status_code}")
        except Exception as e:
            print(f"❌ 서버 통계 조회 실패: {e}")
        return None

    def recognize_sign_sequence(self, features, confidence_threshold=0.6):
        """수어 시퀀스 인식"""
        if not self.recognizer or not self.recognizer.sign_model:
            return {
                'predictions': [],
                'message': 'Sign recognition model not loaded',
                'has_model': False
            }
        
        try:
            return self.recognizer.predict_sign_sequence(features, confidence_threshold)
        except Exception as e:
            print(f"❌ 수어 인식 실패: {e}")
            return {
                'predictions': [],
                'message': f'Recognition failed: {str(e)}',
                'has_model': True
            }

# 사용 예시 및 키보드 단축키 안내
def print_usage():
    """사용법 출력"""
    print("\n" + "="*60)
    print("🎥 Enhanced Webcam with Pose Estimation & Sign Recognition")
    print("="*60)
    print("기본 조작:")
    print("  q: 종료")
    print("  r: 녹화 시작/중지")
    print("  c: 이미지 캡처")
    print("  i: 연속 이미지 저장 시작/중지")
    print("")
    print("포즈 추정:")
    print("  p: 실시간 포즈 추정 토글")
    print("  s: 스켈레톤 표시 토글")
    print("  1~5: 포즈 임계값 설정 (1.0~5.0)")
    print("")
    print("수어 인식:")
    print("  g: 수어 인식 모드 토글")
    print("  d: 수어 예측 표시 토글")
    print("  f: 수어 예측 평활화 토글")
    print("  6~9: 수어 신뢰도 임계값 (0.3~0.9)")
    print("  x: 수어 버퍼 클리어")
    print("")
    print("기타:")
    print("  t: 서버 통계 조회")
    print("  h: 도움말 재출력")
    print("="*60)

if __name__ == "__main__":
    # 웹캠 캡처 객체 생성
    webcam = WebcamCapture()
    
    # 사용법 출력
    print_usage()
    
    try:
        # 카메라 초기화
        if not webcam.initialize_camera():
            print("❌ 카메라 초기화 실패")
            exit(1)
        
        print("✅ 웹캠 시작됨. 키보드로 조작하세요.")
        
        while True:
            frame = webcam.capture_frame()
            if frame is not None:
                cv2.imshow('Enhanced Webcam', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 종료
            if key == ord('q'):
                break
            # 녹화 토글
            elif key == ord('r'):
                if webcam.recording:
                    print(webcam.stop_recording())
                else:
                    print(webcam.start_recording())
            # 이미지 캡처
            elif key == ord('c'):
                webcam.save_image(save_skeleton=webcam.show_skeleton, save_sign=webcam.show_sign_prediction)
                print("📷 이미지 저장됨")
            # 연속 이미지 저장 토글
            elif key == ord('i'):
                if webcam.capturing_images:
                    print(webcam.stop_capture_images())
                else:
                    print(webcam.start_capture_images())
            # 실시간 포즈 추정 토글
            elif key == ord('p'):
                if webcam.realtime_translate:
                    print(webcam.stop_realtime())
                else:
                    print(webcam.start_realtime())
            # 스켈레톤 표시 토글
            elif key == ord('s'):
                print(webcam.toggle_skeleton_display())
            # 수어 인식 모드 토글
            elif key == ord('g'):
                print(webcam.toggle_sign_recognition())
            # 수어 예측 표시 토글
            elif key == ord('d'):
                print(webcam.toggle_sign_prediction_display())
            # 수어 예측 평활화 토글
            elif key == ord('f'):
                print(webcam.toggle_sign_smoothing())
            # 수어 버퍼 클리어
            elif key == ord('x'):
                webcam.clear_sign_buffer()
            # 서버 통계 조회
            elif key == ord('t'):
                webcam.get_server_stats()
            # 도움말
            elif key == ord('h'):
                print_usage()
            # 포즈 임계값 설정 (1~5)
            elif key == ord('1'):
                print(webcam.set_pose_threshold(1.0))
            elif key == ord('2'):
                print(webcam.set_pose_threshold(2.0))
            elif key == ord('3'):
                print(webcam.set_pose_threshold(3.0))
            elif key == ord('4'):
                print(webcam.set_pose_threshold(4.0))
            elif key == ord('5'):
                print(webcam.set_pose_threshold(5.0))
            # 수어 신뢰도 임계값 설정 (6~9)
            elif key == ord('6'):
                print(webcam.set_sign_confidence_threshold(0.3))
            elif key == ord('7'):
                print(webcam.set_sign_confidence_threshold(0.5))
            elif key == ord('8'):
                print(webcam.set_sign_confidence_threshold(0.7))
            elif key == ord('9'):
                print(webcam.set_sign_confidence_threshold(0.9))
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    
    finally:
        # 리소스 정리
        webcam.release_camera()
        cv2.destroyAllWindows()
        print("👋 웹캠 종료됨")