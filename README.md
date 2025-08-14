# Korean Sign Language Recognition System
# 한국 수어 인식 시스템

AIHub 데이터셋을 활용한 한국 수어 인식 및 실시간 번역 시스템입니다. 데이터 전처리부터 모델 훈련, 실시간 서버 배포까지 전체 파이프라인을 제공합니다.

## 🎬 데모 영상

### 실시간 수어 인식 예시

[Click here to view the video](https://github.com/user-attachments/assets/69441a40-d08f-4c6b-b093-d7609ce7eed6)

*실시간 웹캠을 통한 한국 수어 인식 및 번역*

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 구조](#시스템-구조)
- [요구사항](#요구사항)
- [설치 방법](#설치-방법)
- [데이터 준비 및 전처리](#데이터-준비-및-전처리)
- [모델 훈련](#모델-훈련)
- [서버 배포 및 실행](#서버-배포-및-실행)
- [사용 방법](#사용-방법)
- [문제 해결](#문제-해결)
- [기여 방법](#기여-방법)

## 🎯 프로젝트 개요

본 시스템은 청각 장애인과 일반인 간의 의사소통 장벽을 해소하기 위한 AI 기반 한국 수어 인식 및 번역 시스템입니다.

### 주요 특징

#### 🤖 모델 개발 (model 폴더)
- **AIHub 한국 수어 데이터셋** 활용 (536,000 수어영상 클립)
- **OpenHands 모델** 기반 Transformer 아키텍처
- **MediaPipe** 기반 손 및 포즈 랜드마크 추출
- **시퀀스-투-시퀀스** 학습으로 연속 수어 인식
- **Intel GPU 최적화** 지원 (Intel Extension for PyTorch)

#### 🌐 실시간 서버 (server 폴더)
- **실시간 웹캠** 기반 수어 인식 및 번역
- **COCO Wholebody** 133개 키포인트 포즈 추정
- **Flask 기반** 웹 인터페이스 및 REST API
- **YOLOv11** 기반 사람 검출
- **멀티클라이언트** 지원

### 지원 기능

- 수어 단어 및 문장 인식 (2,000개 수어문장, 3,000개 수어단어)
- 실시간 손 랜드마크 추출 (21개 포인트 × 2손 = 42개 포인트)
- 상체 포즈 랜드마크 추출 (6개 주요 포인트)
- 다중 각도 데이터 처리 (5개 각도)
- 다중 화자 지원 (16명 화자)
- 웹 기반 실시간 모니터링 및 제어

## 🏗 시스템 구조

```
korean_sign_language_recognition/
├── model/                          # 모델 개발 및 훈련
│   ├── main.py                     # 전체 파이프라인 실행
│   ├── openhands_finetuner.py      # 모델 정의 및 훈련
│   ├── data_preprocessor.py        # 데이터 전처리
│   ├── test_setup.py               # 설치 환경 테스트
│   ├── requirements-basic.txt      # 기본 패키지
│   ├── requirements.txt            # 전체 패키지
│   ├── aihub_data/                 # AIHub 원본 데이터
│   ├── processed_data/             # 전처리된 데이터
│   └── models/                     # 훈련된 수어 인식 모델
│
├── server/                         # 실시간 서버 및 클라이언트
│   ├── model.pt                    # 포즈 추정 모델
│   ├── enhanced_pose_server.py     # 통합 포즈/수어 서버
│   ├── enhanced_webcam_client.py   # 웹캠 클라이언트
│   ├── flask_web_interface.py      # Flask 웹 인터페이스
│   ├── class_to_idx.py             # 클래스 매핑 추출
│   ├── templates/                  # HTML 템플릿
│   └── configs/                    # MMPose 설정 파일
│
└── README.md
```

### 시스템 아키텍처

#### 데이터 처리 파이프라인
```
AIHub 원본 데이터 → 전처리 → 특징 추출 → 모델 훈련 → 평가 → 배포
```

#### 실시간 서비스 아키텍처
```
웹캠 입력 → YOLOv11 사람검출 → RTMW 포즈추정 → MediaPipe 특징추출 → Transformer 수어인식 → 결과 반환
```

## 🛠 요구사항

### 하드웨어 요구사항

- **메모리**: 최소 16GB RAM (32GB 권장)
- **GPU**: Intel GPU (Arc, Iris Xe) 또는 NVIDIA GPU
- **저장공간**: 최소 100GB (전체 데이터셋 + 모델)
- **웹캠**: USB 카메라 또는 내장 카메라 (서버 실행 시)

### 소프트웨어 요구사항

- **Python**: 3.8 이상 (3.9 권장)
- **운영체제**: Linux, Windows, macOS
- **Intel GPU 드라이버**: 최신 버전 (Intel GPU 사용 시)

### 핵심 라이브러리

#### 모델 개발용
- PyTorch (Intel XPU 지원)
- MediaPipe
- OpenCV
- NumPy, Pandas
- Transformers

#### 서버 배포용
- Flask
- MMPose
- Ultralytics (YOLOv11)
- Intel Extension for PyTorch

## 📦 설치 방법

### 1. 저장소 클론 및 기본 설정

```bash
# 저장소 클론
git clone <repository-url>
cd korean-sign-language-recognition

# Python 환경 설정
conda create -n korean-sign python=3.9
conda activate korean-sign

# 또는 venv 사용
python -m venv korean-sign
source korean-sign/bin/activate  # Linux/Mac
# korean-sign\Scripts\activate  # Windows
```

### 2. 모델 개발 환경 설치

```bash
cd model

# 기본 패키지 설치 (권장)
pip install -r requirements-basic.txt

# 전체 패키지 설치 (선택)
pip install -r requirements.txt

# Intel GPU 지원 (선택)
chmod +x install_intel_gpu.sh
./install_intel_gpu.sh

# 설치 확인
python test_setup.py
```

### 3. 서버 환경 설치

```bash
cd ../server

# 서버 패키지 설치
pip install -r requirements.txt

# MMPose 설치
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmpose

# Intel GPU 지원 (선택)
pip install intel-extension-for-pytorch
```

### 4. 모델 다운로드

```bash
cd server
mkdir -p models configs/wholebody_2d_keypoint/rtmpose/cocktail14

# RTMW 포즈 추정 모델
wget -P models/ \
  https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288-20231122.pth

# MMPose 설정 파일
wget -P configs/wholebody_2d_keypoint/rtmpose/cocktail14/ \
  https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py
```

## 📊 데이터 준비 및 전처리

### AIHub 데이터셋 구조

AIHub 한국 수어 데이터셋을 다음과 같이 배치하세요:

```
model/aihub_data/
├── sentence_001/
│   ├── NIA_SL_SEN0001_REAL01_F.mp4      # 정면
│   ├── NIA_SL_SEN0001_REAL01_F_morpheme.json
│   ├── NIA_SL_SEN0001_REAL01_L.mp4      # 좌측
│   ├── NIA_SL_SEN0001_REAL01_L_morpheme.json
│   ├── NIA_SL_SEN0001_REAL01_R.mp4      # 우측
│   └── ... (다른 각도 및 화자)
├── sentence_002/
└── ...
```

### 데이터 전처리 실행

```bash
cd model

# 데이터 전처리
python main.py preprocess \
    --data_dir ./aihub_data \
    --output_dir ./processed_data \
    --sequence_length 32 \
    --train_ratio 0.8
```

**전처리 과정:**
- MP4 비디오에서 프레임 추출
- MediaPipe로 손 랜드마크 추출 (126차원)
- MediaPipe로 포즈 랜드마크 추출 (18차원)
- JSON 어노테이션과 동기화
- 시퀀스 데이터 생성 및 분할 (train/validation)

### JSON 어노테이션 형식

```json
{
    "metaData": {
        "name": "NIA_SL_SEN0001_REAL01_F.mp4",
        "duration": 3.25
    },
    "data": [
        {
            "start": 1.422,
            "end": 2.484,
            "attributes": [{"name": "나"}]
        }
    ]
}
```

## 🚀 모델 훈련

### 1. 기본 훈련

```bash
cd model

# 기본 설정으로 훈련
python main.py train \
    --processed_data_dir ./processed_data \
    --model_save_dir ./trained_models \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50
```

### 2. 전체 파이프라인 실행

```bash
# 전처리 + 훈련을 한 번에
python main.py pipeline \
    --data_dir ./aihub_data \
    --output_dir ./processed_data \
    --model_save_dir ./trained_models \
    --num_epochs 100 \
    --batch_size 32
```

### 3. 고급 하이퍼파라미터 조정

```bash
python main.py train \
    --processed_data_dir ./processed_data \
    --model_save_dir ./trained_models \
    --d_model 512 \        # 모델 차원 증가
    --n_heads 16 \         # 어텐션 헤드 증가
    --n_layers 12 \        # 레이어 수 증가
    --batch_size 8 \       # 메모리 부족 시 감소
    --learning_rate 5e-5   # 학습률 조정
```

### 모델 구조

```
입력: (batch_size, seq_len, 144)  # MediaPipe 특징
  ↓
Feature Projection: Linear(144 → d_model)
  ↓
Positional Encoding: Learnable positional embeddings
  ↓
Transformer Encoder: 
  - Multi-Head Attention (n_heads)
  - Feed Forward Network
  - Layer Normalization
  - Dropout (×n_layers)
  ↓
Classification Head: Linear(d_model → vocab_size)
  ↓
출력: (batch_size, seq_len, vocab_size)  # 수어 단어 분류
```

## 🌐 서버 배포 및 실행

### 1. 클래스 매핑 설정

```bash
cd server

# 훈련된 모델에서 클래스 정보 추출
python class_to_idx.py ../model/trained_models/best_model.pt

# 출력된 클래스 정보를 enhanced_pose_server.py에 추가
```

### 2. 통합 서버 실행

```bash
# 포즈 추정 + 수어 인식 서버 실행
python enhanced_pose_server.py \
  --config ../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
  --checkpoint ../models/rtmw-l_8xb320-270e_cocktail14-384x288-20231122.pth \
  --sign-model ../model/trained_models/best_model.pt \
  --device auto \
  --yolo-model n \
  --port 5000
```

### 3. 클라이언트 실행

#### 콘솔 클라이언트
```bash
# 콘솔 버전 실행
python enhanced_webcam_client.py
```

#### 웹 인터페이스
```bash
# 웹 인터페이스 실행
python flask_web_interface.py

# 웹 브라우저에서 접속
# http://localhost:8000
```

## 🎮 사용 방법

### 콘솔 클라이언트 키보드 단축키

#### 기본 조작
- `q`: 종료
- `r`: 녹화 시작/중지
- `c`: 이미지 캡처
- `h`: 도움말 표시

#### 포즈 추정 제어
- `p`: 실시간 포즈 추정 토글
- `s`: 스켈레톤 표시 토글
- `1~5`: 포즈 임계값 설정 (1.0~5.0)

#### 수어 인식 제어
- `g`: 수어 인식 모드 토글
- `d`: 수어 예측 표시 토글
- `f`: 수어 예측 평활화 토글
- `6~9`: 수어 신뢰도 임계값 (0.3~0.9)
- `x`: 수어 버퍼 클리어

#### 모니터링
- `t`: 서버 통계 조회

### 웹 인터페이스 기능

1. **실시간 비디오 스트림**: 웹캠 영상과 분석 결과
2. **포즈 추정 제어**: 포즈 추정, 스켈레톤 표시, 임계값 조절
3. **수어 인식 제어**: 수어 인식 모드, 결과 표시, 평활화
4. **녹화 및 캡처**: 비디오 녹화, 이미지 캡처
5. **서버 모니터링**: 실시간 통계 및 상태 확인

### API 엔드포인트

#### 포즈 추정
- `POST /estimate_pose`: 이미지 → 포즈 키포인트
- `GET /health`: 서버 상태 확인
- `GET /stats`: 서버 통계

#### 수어 인식
- `POST /sign_recognition`: 통합 수어 인식
- `POST /extract_sign_features`: MediaPipe 특징 추출
- `POST /predict_sign`: 버퍼된 특징으로 수어 예측
- `POST /clear_buffer/<client_id>`: 클라이언트별 버퍼 클리어

## 📈 성능 및 평가

### 모델 성능 지표

#### 예상 성능
- **어휘 크기**: 500-1000개 수어 단어
- **시퀀스 정확도**: 85-92% (데이터셋 품질에 따라)
- **Token-level Accuracy**: 개별 수어 단어 정확도
- **BLEU Score**: 시퀀스 유사도 측정

#### 실시간 처리 성능
- **실시간 처리**: 30 FPS (Intel GPU)
- **지연시간**: < 100ms (로컬 처리)
- **메모리 사용량**: 4-8GB (배치 크기에 따라)

### 하드웨어별 성능 비교

| 하드웨어 | YOLOv11 FPS | 포즈 추정 FPS | 수어 인식 지연시간 |
|----------|-------------|---------------|-------------------|
| Intel Arc A770 | 45-60 | 35-40 | ~50ms |
| NVIDIA RTX 3070 | 50-70 | 40-45 | ~40ms |
| Intel i7 CPU | 15-25 | 10-15 | ~200ms |

### 모니터링 및 로깅

```bash
# TensorBoard 실행 (훈련 시)
cd model
tensorboard --logdir ./trained_models/logs

# 실시간 서버 로그
tail -f korean_sign_recognition.log
```

## 🔧 문제 해결

### 모델 훈련 관련

#### Intel GPU 인식 실패
```bash
# Intel GPU 드라이버 확인
intel_gpu_top

# OneAPI 환경 설정
source /opt/intel/oneapi/setvars.sh

# Python에서 확인
python -c "import torch; print('XPU available:', torch.xpu.is_available())"
```

#### 메모리 부족
```bash
# 배치 크기 감소
python main.py train --batch_size 8

# 시퀀스 길이 감소
python main.py train --sequence_length 16

# 모델 크기 감소
python main.py train --d_model 128 --n_layers 4
```

#### MediaPipe 설치 오류
```bash
# 시스템 의존성 설치 (Ubuntu)
sudo apt-get install libgl1-mesa-glx

# macOS
brew install opencv

# MediaPipe 재설치
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

### 서버 실행 관련

#### 카메라 접근 오류
```python
# 카메라 인덱스 변경
self.cap = cv2.VideoCapture(1)  # 0 → 1로 변경
```

#### 서버 연결 실패
```bash
# 방화벽 설정 확인
sudo ufw allow 5000

# 다른 포트 사용
python enhanced_pose_server.py --port 5001
```

#### 성능 최적화
```python
# FPS 조절
self.fps = 5  # 낮은 FPS로 설정

# 임계값 조절
self.pose_threshold = 3.0  # 높은 임계값
```

### GPU 관련 문제

#### CUDA vs Intel GPU 충돌
```bash
# CUDA 비활성화
export CUDA_VISIBLE_DEVICES=""

# Intel GPU 강제 사용
export USE_INTEL_GPU=1
```

#### 메모리 모니터링
```bash
# Intel GPU 모니터링
watch -n 1 'intel_gpu_top'

# NVIDIA GPU 모니터링
watch -n 1 'nvidia-smi'
```

## 🚀 고급 설정 및 최적화

### Intel GPU 최적화

```bash
# 환경 변수 설정
export USE_INTEL_GPU=1

# 코드에서 최적화 활성화
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
```

### 데이터 증강 및 전처리 최적화

```python
# 고급 전처리 옵션
python main.py preprocess \
    --data_dir ./aihub_data \
    --output_dir ./processed_data \
    --sequence_length 64 \      # 더 긴 시퀀스
    --overlap_ratio 0.5 \       # 오버랩 증가
    --augment_data \           # 데이터 증강 활성화
    --normalize_landmarks      # 랜드마크 정규화
```

### 분산 훈련 설정

```bash
# 다중 GPU 훈련
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py train \
    --distributed \
    --batch_size 32
```

## 📁 출력 파일 구조

```
korean-sign-language-recognition/
├── model/
│   ├── processed_data/
│   │   ├── train_data.pt           # 훈련 데이터
│   │   ├── val_data.pt             # 검증 데이터
│   │   └── vocab.json              # 어휘 사전
│   ├── trained_models/
│   │   ├── best_model.pt           # 최고 성능 모델
│   │   ├── final_model.pt          # 최종 모델
│   │   ├── training_history.json   # 훈련 이력
│   │   └── logs/                   # TensorBoard 로그
│   └── korean_sign_recognition.log # 모델 훈련 로그 파일
│
└── server/
    ├── model.pt                    # 포즈 추정 모델
    ├── captured_videos/            # 녹화된 비디오
    ├── captured_images/            # 캡처된 이미지
    └── captured_datas/             # 연속 캡처 데이터
```

## 📋 체크리스트

### 설치 확인
- [ ] Python 3.8+ 설치
- [ ] conda/venv 환경 생성
- [ ] model 패키지 설치 완료
- [ ] server 패키지 설치 완료
- [ ] Intel GPU/CUDA 설정 (선택)

### 데이터 준비
- [ ] AIHub 데이터셋 다운로드
- [ ] 데이터 구조 확인
- [ ] 전처리 실행 완료
- [ ] train/val 데이터 생성 확인

### 모델 훈련
- [ ] 기본 훈련 실행 성공
- [ ] best_model.pt 생성 확인
- [ ] 훈련 로그 확인
- [ ] 성능 지표 확인

### 서버 배포
- [ ] RTMW 모델 다운로드
- [ ] MMPose 설정 완료
- [ ] 클래스 매핑 설정
- [ ] 서버 실행 성공
- [ ] 웹캠 연결 확인

### 테스트
- [ ] 콘솔 클라이언트 동작 확인
- [ ] 웹 인터페이스 접속 확인
- [ ] 실시간 수어 인식 테스트
- [ ] API 엔드포인트 테스트

## 🤝 기여 방법

1. **이슈 리포트**: 버그나 개선사항 제안
2. **코드 기여**: Pull Request 환영
3. **데이터셋 기여**: 추가 수어 데이터 제공
4. **문서 개선**: README나 주석 개선
5. **성능 최적화**: 알고리즘 개선 제안

### 개발 가이드라인

- 코드 스타일: PEP 8 준수
- 커밋 메시지: Conventional Commits 형식
- 테스트: 주요 기능에 대한 단위 테스트 포함
- 문서: 새로운 기능에 대한 문서 업데이트

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다.

### 데이터셋 라이선스

- **AI Hub 수어 영상 데이터셋**
  - 제공: 한국지능정보사회진흥원 (NIA)
  - 연도: 2021
  - 용도: 연구 및 교육 목적
  - 상업적 사용 시 별도 라이선스 확인 필요

## 📞 지원 및 연락처

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **이메일**: [개발자 이메일]
- **위키**: [프로젝트 위키 링크]
- **데모**: [온라인 데모 링크]

## 📚 참고문헌

1. AI Hub 수어 영상 데이터셋 (한국지능정보사회진흥원, 2021)
2. OpenHands: Making Sign Language Recognition Accessible (arXiv preprint)
3. MediaPipe Hands: On-device Real-time Hand Tracking (Google AI)
4. Real-Time Multi-Person 2D Pose Estimation using Part Affinity Fields
5. YOLOv11: An Improved Version of YOLO for Object Detection

---

**주의사항**: 본 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 서비스 배포 시에는 성능 최적화 및 보안 강화가 필요할 수 있습니다.
