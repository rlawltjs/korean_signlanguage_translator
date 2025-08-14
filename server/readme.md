# Korean Sign Language Translator

한국 수어를 텍스트로 번역하는 AI 기반 실시간 번역 시스템입니다.

## 프로젝트 개요

이 프로젝트는 웹캠을 통해 실시간으로 한국 수어를 인식하고 텍스트로 번역하는 시스템입니다. COCO Wholebody 포즈 추정과 딥러닝 기반 수어 인식 모델을 결합하여 청각 장애인과 일반인 간의 의사소통 장벽을 해소하는 것을 목표로 합니다.

## 주요 기능

### 🦴 포즈 추정
- 실시간 COCO Wholebody 133개 키포인트 추정
- 전신, 손, 얼굴 포즈 동시 인식
- 스켈레톤 시각화 및 임계값 조절

### 🤟 수어 인식
- 실시간 수어 동작 인식 및 번역
- 수어 단어 및 문장 인식
- 신뢰도 기반 예측 결과 필터링
- 시계열 데이터 평활화를 통한 정확도 향상

### 📹 웹캠 제어
- 실시간 비디오 스트리밍
- 녹화 및 이미지 캡처 기능
- 연속 이미지 저장 및 처리

### 🌐 웹 인터페이스
- Flask 기반 실시간 웹 인터페이스
- 직관적인 제어 패널
- 실시간 서버 통계 및 상태 모니터링

## 데이터셋

본 프로젝트는 AI Hub에서 제공하는 수어 영상 데이터셋을 활용합니다.

### 데이터 출처

* **AI Hub**
   * 이름: 수어 영상
   * 제공: 한국지능정보사회진흥원
   * 연도: 2021
   * 링크: www.aihub.or.kr (데이터셋 번호: 103)
   * 구성:
     * 총 536,000 수어영상 클립 (`.mp4`)
     * 수어문장 2000개, 수어단어 3000개
     * 지숫자/지문자 1000개

## 시스템 구조

### 클라이언트-서버 아키텍처
- **클라이언트**: 웹캠 캡처 및 실시간 처리 (`enhanced_webcam_client.py`)
- **웹 인터페이스**: Flask 기반 웹 제어 패널 (`flask_web_interface.py`)
- **서버**: 포즈 추정 및 수어 인식 통합 서버 (`enhanced_pose_server.py`)

### 핵심 컴포넌트

#### 포즈 추정 파이프라인
- **PersonDetector**: YOLOv11 기반 사람 1명 검출 (Intel XPU 지원)
- **RTMWPoseEstimator**: RTMW 모델 기반 133개 키포인트 추정
- **COCO Wholebody**: 전신(17) + 손(42) + 얼굴(68) + 발(6) 키포인트

#### 수어 인식 파이프라인
- **MediaPipe**: 실시간 손/포즈 랜드마크 추출
- **OpenHandsTransformerModel**: Transformer 기반 시계열 분류 모델
- **SignLanguageRecognizer**: 특징 추출 → 시퀀스 버퍼링 → 수어 예측

#### 통합 서버 시스템
- **PoseServer**: Flask 기반 REST API 서버
- **멀티 엔드포인트**: 포즈 추정, 수어 인식, 특징 추출 지원
- **클라이언트별 버퍼 관리**: 실시간 다중 클라이언트 지원

## 시스템 요구사항

### 기본 요구사항
- Python 3.8+
- OpenCV 4.x
- Flask 2.x
- NumPy
- PyTorch (XPU/CUDA 지원)
- MMPose
- Ultralytics (YOLOv11)
- MediaPipe
- Transformers (OpenHands 모델용)

### 하드웨어 요구사항
- 웹캠 (USB 카메라 또는 내장 카메라)
- **권장**: Intel Arc GPU (XPU 지원) 또는 NVIDIA GPU (CUDA)
- **최소**: CPU (Intel/AMD)
- 최소 8GB RAM

### GPU 지원
- **Intel Arc GPU (XPU)**: 최우선 지원, Intel Extension for PyTorch 필요
- **NVIDIA GPU (CUDA)**: 완전 지원
- **CPU 폴백**: 모든 기능 지원 (속도 저하)

## 설치 및 실행

### 1. 저장소 클론 및 환경 설정

```bash
# 저장소 클론
git clone https://github.com/rlawltjs/korean_signlanguage_translator.git
cd korean_signlanguage_translator

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 기본 패키지 설치
pip install -r requirements.txt

# Intel Arc GPU 사용자 (선택사항)
pip install intel-extension-for-pytorch

# MMPose 설치
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmpose
```

### 2. 모델 다운로드

```bash
# RTMW 포즈 추정 모델 다운로드
mkdir -p models
cd models

# RTMW-L 모델 (권장)
wget https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288-20231122.pth

cd ..

# MMPose 설정 파일 다운로드
mkdir -p configs/wholebody_2d_keypoint/rtmpose/cocktail14
wget -P configs/wholebody_2d_keypoint/rtmpose/cocktail14/ \
  https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py
```

### 3. 서버 실행

```bash
cd server
python enhanced_pose_server.py \
  --config ../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
  --checkpoint ../models/rtmw-l_8xb320-270e_cocktail14-384x288-20231122.pth \
  --device auto \
  --yolo-model n \
  --port 5000
```

### 4. 클라이언트 실행

#### 콘솔 버전
```bash
cd server
python enhanced_webcam_client.py
```

#### 웹 인터페이스
```bash
cd server
python flask_web_interface.py
```
웹 브라우저에서 `http://localhost:8000` 접속

### 5. 수어 인식 모델 (선택사항)

OpenHands Transformer 수어 인식 모델을 사용하려면:

```bash
# 서버 실행 시 수어 모델 경로 추가
python enhanced_pose_server.py \
  --sign-model path/to/your/sign_model.pt \
  [다른 옵션들...]
```

## 사용법

### 콘솔 버전 키보드 단축키

#### 기본 조작
- `q`: 종료
- `r`: 녹화 시작/중지
- `c`: 이미지 캡처
- `i`: 연속 이미지 저장 시작/중지
- `h`: 도움말 표시

#### 포즈 추정
- `p`: 실시간 포즈 추정 토글
- `s`: 스켈레톤 표시 토글
- `1~5`: 포즈 임계값 설정 (1.0~5.0)

#### 수어 인식
- `g`: 수어 인식 모드 토글
- `d`: 수어 예측 표시 토글
- `f`: 수어 예측 평활화 토글
- `6~9`: 수어 신뢰도 임계값 (0.3~0.9)
- `x`: 수어 버퍼 클리어

#### 기타
- `t`: 서버 통계 조회

### 웹 인터페이스

웹 인터페이스를 통해 다음 기능들을 제어할 수 있습니다:

1. **실시간 비디오 스트림**: 웹캠 영상과 실시간 분석 결과 확인
2. **포즈 추정 제어**: 포즈 추정 시작/중지, 스켈레톤 표시, 임계값 조절
3. **수어 인식 제어**: 수어 인식 모드, 결과 표시, 평활화 설정
4. **녹화 및 캡처**: 비디오 녹화, 연속 이미지 캡처
5. **서버 모니터링**: 실시간 서버 통계 및 상태 확인

## 모델 구조

### YOLOv11 사람 검출
- **PersonDetector**: YOLOv11 기반 사람 1명 검출 (최고 신뢰도)
- **성능 향상**: YOLOv8 대비 22% 적은 파라미터로 더 높은 mAP
- **속도 개선**: YOLOv8 대비 평균 40% 빠른 추론 속도
- **모델 옵션**: nano(n), small(s), medium(m), large(l), xlarge(x)
- **Intel XPU 지원**: Intel Arc GPU 최적화

### RTMW 포즈 추정
- **133개 키포인트**: COCO Wholebody 표준
  - 전신: 17개 (기본 스켈레톤)
  - 손: 42개 (양손 각 21개)
  - 얼굴: 68개 (얼굴 랜드마크)
  - 발: 6개 (발가락)
- **입력 크기**: 288x384 (최적화된 해상도)
- **좌표 변환**: 크롭 영역 → 원본 이미지 좌표

### OpenHands Transformer 수어 인식
- **구조**: Transformer Encoder 기반 시계열 분류
- **입력 차원**: 144차원 (MediaPipe 손+포즈 특징)
- **시퀀스 길이**: 32프레임 (약 3초)
- **출력**: 한국어 수어 단어 분류
- **특징**:
  - 위치 인코딩으로 시간 정보 반영
  - 어텐션 마스크로 유효 프레임만 처리
  - Global average pooling으로 시퀀스 집약

### MediaPipe 특징 추출
- **손 랜드마크**: 양손 각 21개 × 3차원 (x,y,z) = 126차원
- **포즈 랜드마크**: 상체 주요 부위 6개 × 3차원 = 18차원
- **총 144차원**: 손 126차원 + 포즈 18차원
- **실시간 처리**: 단일 이미지에서 즉시 특징 추출

## 기술적 특징

### 실시간 처리
- **다중 모드 동시 실행**: 포즈 추정과 수어 인식 동시 처리 가능
- **비동기 통신**: 클라이언트-서버 간 비차단 통신
- **프레임 드롭 방지**: 적응적 FPS 조절
- **멀티스레드 서버**: Flask 멀티스레드 지원으로 다중 클라이언트 처리

### 성능 최적화
- **YOLOv11 성능**: YOLOv8 대비 22% 적은 파라미터, 40% 빠른 추론
- **Intel XPU 최적화**: Intel Arc GPU 우선 지원
- **자동 디바이스 선택**: XPU → CUDA → CPU 순서로 최적 디바이스 자동 선택
- **모델 크기 선택**: nano(3.3M) ~ xlarge(68.2M) 파라미터 모델 지원

### 신뢰도 관리
- **포즈 임계값**: 키포인트 신뢰도 기반 필터링 (1.0~5.0)
- **수어 신뢰도**: 예측 결과 신뢰도 임계값 (0.1~1.0)
- **시계열 평활화**: 연속 프레임 결과 평활화로 정확도 향상
- **어텐션 마스크**: 유효한 프레임만 처리하여 정확도 향상

### 확장성 및 유지보수
- **클라이언트별 버퍼 관리**: 개별 클라이언트 특징 시퀀스 관리
- **RESTful API**: 표준 HTTP API로 다양한 클라이언트 지원
- **모듈형 설계**: 포즈 추정, 수어 인식 독립적 사용 가능
- **실시간 모니터링**: 서버 통계 및 성능 메트릭 실시간 조회

## 프로젝트 구조

```
korean_signlanguage_translator/
├── server/                              # 서버 및 클라이언트 코드
│   ├── enhanced_pose_server.py          # 통합 포즈/수어 서버 (메인 서버)
│   ├── enhanced_webcam_client.py        # 웹캠 클라이언트 (콘솔 버전)
│   ├── flask_web_interface.py           # Flask 웹 인터페이스
│   └── templates/                       # HTML 템플릿
│       └── index.html                   # 웹 인터페이스 메인 페이지
├── models/                              # 학습된 모델 저장
│   └── rtmw-l_*.pth                    # RTMW 포즈 추정 모델
├── configs/                            # MMPose 설정 파일
│   └── wholebody_2d_keypoint/
│       └── rtmpose/
│           └── cocktail14/
│               └── rtmw-l_*.py         # RTMW 설정 파일
├── captured_videos/                     # 녹화된 비디오 저장
├── captured_images/                     # 캡처된 이미지 저장
├── captured_datas/                      # 연속 캡처 데이터
├── requirements.txt                     # 필요 패키지 목록
├── README.md
└── sign_model.pt                   # 수어 인식 모델 (선택)
```

## 서버 API 엔드포인트

### 포즈 추정
- `POST /estimate_pose`: 이미지 → 포즈 키포인트 반환
- `GET /health`: 서버 상태 확인
- `GET /stats`: 서버 통계 조회

### 수어 인식
- `POST /sign_recognition`: 통합 수어 인식 (특징 추출 + 예측)
- `POST /extract_sign_features`: MediaPipe 특징 추출만
- `POST /predict_sign`: 버퍼된 특징으로 수어 예측
- `POST /clear_buffer/<client_id>`: 클라이언트별 특징 버퍼 클리어

## 주요 클래스 및 함수

### 서버 측 (enhanced_pose_server.py)
- **PersonDetector**: YOLOv11 기반 사람 검출
- **RTMWPoseEstimator**: RTMW 포즈 추정
- **OpenHandsTransformerModel**: Transformer 기반 수어 분류 모델
- **SignLanguageRecognizer**: MediaPipe + Transformer 수어 인식
- **PoseServer**: Flask 기반 통합 서버

### 클라이언트 측 (enhanced_webcam_client.py)
- **WebcamCapture**: 웹캠 제어 및 실시간 처리
- `toggle_skeleton_display()`: 스켈레톤 표시 토글
- `toggle_sign_recognition()`: 수어 인식 모드 토글
- `send_frame_for_pose_estimation()`: 포즈 추정 요청
- `send_frame_for_sign_recognition()`: 수어 인식 요청
- `smooth_sign_predictions()`: 수어 예측 결과 평활화

### 시각화 함수
- `COCO_WHOLEBODY_SKELETON`: 133개 키포인트 연결 정보
- `draw_keypoints_wholebody_on_frame()`: 스켈레톤 시각화
- `draw_sign_prediction_on_frame()`: 수어 예측 결과 시각화

## 설정 및 커스터마이징

### 서버 설정
```python
# enhanced_pose_server.py 실행 옵션
python enhanced_pose_server.py \
  --device auto \           # 디바이스 선택 (auto/cpu/cuda/xpu)
  --yolo-model n \          # YOLO 모델 크기 (n/s/m/l/x)
  --port 5000 \            # 서버 포트
  --detection-conf 0.5 \   # 사람 검출 신뢰도 임계값
  --sign-model path.pt     # 수어 모델 경로 (선택사항)
```

### 클라이언트 설정
```python
# enhanced_webcam_client.py에서 서버 URL 변경
server_url = "http://000.000.000.000:5000"  # 실제 서버 IP로 변경

# 카메라 설정
self.w = 640          # 웹캠 너비
self.h = 480          # 웹캠 높이
self.fps = 10         # FPS
```

### 수어 인식 파라미터
```python
# 수어 인식 모델 설정
class OpenHandsTransformerModel:
    input_size = 144           # MediaPipe 특징 차원
    hidden_size = 256          # Transformer hidden 차원
    num_classes = 14           # 수어 클래스 수
    sequence_length = 32       # 시퀀스 길이 (프레임)
    
# 클라이언트 설정
self.sign_prediction_buffer = deque(maxlen=5)    # 평활화 버퍼 크기
self.sign_confidence_threshold = 0.6             # 기본 신뢰도 임계값
```

### Intel Arc GPU (XPU) 최적화
```bash
# Intel Extension for PyTorch 설치
pip install intel-extension-for-pytorch

# XPU 디바이스 확인
python -c "import torch; print(torch.xpu.is_available())"
```

## 지원되는 수어 클래스

현재 구현된 한국어 수어 단어들:

| 클래스 ID | 수어 단어 | 설명 |
|----------|----------|------|
| 0 | `<PAD>` | 패딩 토큰 |
| 1 | `<UNK>` | 미지 토큰 |
| 2 | `<SOS>` | 시작 토큰 |
| 3 | `<EOS>` | 종료 토큰 |
| 4 | ...

*참고: 실제 수어 모델에 따라 클래스가 다를 수 있습니다.*

## 문제 해결

### 자주 발생하는 문제

1. **카메라 접근 오류**
   ```bash
   # 카메라 권한 확인 및 다른 앱에서 사용 중인지 확인
   # 카메라 인덱스 변경 (0 → 1)
   self.cap = cv2.VideoCapture(1)
   ```

2. **서버 연결 실패**
   ```bash
   # 서버 IP 주소 확인
   # 방화벽 설정 확인
   # 서버가 실행 중인지 확인
   ```

3. **성능 최적화**
   ```python
   # FPS 조절로 처리 속도 최적화
   self.fps = 5  # 낮은 FPS로 설정
   
   # 임계값 조절로 노이즈 제거
   self.pose_threshold = 3.0  # 높은 임계값
   ```

### 디버깅 팁

- 콘솔 출력으로 실시간 상태 확인
- `t` 키로 서버 통계 모니터링
- 웹 인터페이스에서 시각적 상태 확인

## 연락처

프로젝트에 대한 문의사항이나 버그 리포트는 GitHub Issues를 통해 남겨주세요.

## 참고문헌

- AI Hub 수어 영상 데이터셋 (한국지능정보사회진흥원, 2021)
- 관련 연구 논문 및 참고 자료들
