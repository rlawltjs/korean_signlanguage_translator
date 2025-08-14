# 한국 수어 인식 시스템 (Korean Sign Language Recognition)

AIHub 데이터셋을 활용한 OpenHands 모델 기반 한국 수어 인식 시스템입니다. Intel GPU 지원으로 최적화되어 있습니다.

## 📋 목차

- [시스템 개요](#시스템-개요)
- [요구사항](#요구사항)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [사용 방법](#사용-방법)
- [모델 구조](#모델-구조)
- [성능 및 평가](#성능-및-평가)
- [문제 해결](#문제-해결)

## 🎯 시스템 개요

본 시스템은 다음과 같은 특징을 가집니다:

- **AIHub 한국 수어 데이터셋** 활용
- **OpenHands 모델** 기반 Transformer 아키텍처
- **Intel GPU 최적화** 지원 (Intel Extension for PyTorch)
- **MediaPipe** 기반 실시간 손 및 포즈 랜드마크 추출
- **시퀀스-투-시퀀스** 학습으로 연속 수어 인식

### 지원 기능

- 실시간 손 랜드마크 추출 (21개 포인트 × 2손 = 42개 포인트)
- 상체 포즈 랜드마크 추출 (6개 주요 포인트)
- 시퀀스 기반 수어 단어 인식
- 다중 각도 데이터 처리 (5개 각도)
- 다중 화자 지원 (16명 화자)

## 🛠 요구사항

### 하드웨어 요구사항

- **메모리**: 최소 16GB RAM (32GB 권장)
- **GPU**: Intel GPU (Arc, Iris Xe) 또는 NVIDIA GPU
- **저장공간**: 최소 50GB (데이터셋 + 모델)

### 소프트웨어 요구사항

- **Python**: 3.8 이상
- **운영체제**: Linux, Windows, macOS
- **Intel GPU 드라이버**: 최신 버전 (Intel GPU 사용 시)

## 📦 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd korean-sign-language-recognition
```

### 2. Python 환경 설정

```bash
# conda 환경 생성 (권장)
conda create -n korean-sign python=3.9
conda activate korean-sign

# 또는 venv 사용
python -m venv korean-sign
source korean-sign/bin/activate  # Linux/Mac
# korean-sign\Scripts\activate  # Windows
```

### 3. 기본 패키지 설치

**권장 방법 (가장 안전):**
```bash
# 기본 패키지만 설치 (CPU/CUDA 지원)
pip install -r requirements-basic.txt
```

**또는 전체 패키지 설치:**
```bash
# 모든 패키지 설치
pip install -r requirements.txt
```

### 4. Intel GPU 지원 (선택사항)

Intel GPU를 사용하려면:

```bash
# 자동 설치 스크립트 사용
chmod +x install_intel_gpu.sh
./install_intel_gpu.sh

# 또는 수동 설치
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 5. 설치 확인

```bash
# 시스템 설정 테스트
python test_setup.py
```

이 테스트는 다음을 확인합니다:
- PyTorch 설치 상태
- Intel GPU 지원 여부  
- 필수 의존성 패키지
- MediaPipe 초기화
- 모델 컴포넌트 동작

### 빠른 설치 (한 번에)

```bash
# 1. 기본 패키지 설치
pip install -r requirements-basic.txt

# 2. 설치 확인
python test_setup.py

# 3. 시스템 테스트
python -c "from openhands_finetuner import setup_device; print('Device:', setup_device())"
```

### 문제 해결

#### Intel GPU 호환성 오류
```bash
# PyTorch 2.8.0+와 Intel Extension 호환성 문제 시
pip uninstall intel-extension-for-pytorch
# Intel GPU 없이 진행하거나 호환 버전으로 다운그레이드
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

## 📊 데이터 준비

### AIHub 데이터셋 구조

AIHub 한국 수어 데이터셋은 다음과 같은 구조를 가져야 합니다:

```
aihub_data/
├── sentence_001/
│   ├── NIA_SL_SEN0001_REAL01_F.mp4
│   ├── NIA_SL_SEN0001_REAL01_F_morpheme.json
│   ├── NIA_SL_SEN0001_REAL01_L.mp4
│   ├── NIA_SL_SEN0001_REAL01_L_morpheme.json
│   ├── NIA_SL_SEN0001_REAL01_R.mp4
│   ├── NIA_SL_SEN0001_REAL01_R_morpheme.json
│   └── ... (다른 각도 및 화자)
├── sentence_002/
│   └── ... (동일 구조)
└── ...
```

### JSON 어노테이션 형식

각 비디오에 대응하는 JSON 파일은 다음과 같은 구조를 가집니다:

```json
{
    "metaData": {
        "url": "비디오 URL",
        "name": "파일명.mp4",
        "duration": 3.25,
        "exportedOn": "2020/12/10"
    },
    "data": [
        {
            "start": 1.422,
            "end": 2.484,
            "attributes": [
                {
                    "name": "나"
                }
            ]
        }
    ]
}
```

## 🚀 사용 방법

### 1. 빠른 시작 (전체 파이프라인)

```bash
# 전체 파이프라인 실행 (전처리 + 훈련)
python main.py pipeline \
    --data_dir ./aihub_data \
    --output_dir ./processed_data \
    --model_save_dir ./models \
    --num_epochs 50 \
    --batch_size 16
```

### 2. 단계별 실행

#### 2.1 데이터 전처리

```bash
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
- 시퀀스 데이터 생성 및 분할

#### 2.2 모델 훈련

```bash
python main.py train \
    --processed_data_dir ./processed_data \
    --model_save_dir ./models \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 6
```

**훈련 과정:**
- Transformer 기반 시퀀스-투-시퀀스 학습
- AdamW 옵티마이저 + 학습률 스케줄링
- Early stopping (patience=10)
- 최고 성능 모델 자동 저장

#### 2.3 추론 테스트

```bash
python main.py inference \
    --model_path ./models/best_model.pt \
    --processed_data_dir ./processed_data
```

### 3. 고급 설정

#### 모델 하이퍼파라미터 조정

```bash
python main.py train \
    --processed_data_dir ./processed_data \
    --model_save_dir ./models \
    --d_model 512 \        # 모델 차원 증가
    --n_heads 16 \         # 어텐션 헤드 증가
    --n_layers 12 \        # 레이어 수 증가
    --batch_size 8 \       # 메모리 부족 시 배치 크기 감소
    --learning_rate 5e-5   # 학습률 조정
```

#### Intel GPU 최적화 활성화

```bash
# 환경 변수 설정
export USE_INTEL_GPU=1

# Intel Extension for PyTorch 확인
python -c "import intel_extension_for_pytorch as ipex; print('Intel XPU available:', ipex.xpu.is_available())"
```

## 🏗 모델 구조

### OpenHands Korean Sign Model

```
입력: (batch_size, seq_len, 144)
  ↓
Feature Projection: Linear(144 → d_model)
  ↓
Positional Encoding: Learnable positional embeddings
  ↓
Transformer Encoder: 
  - Multi-Head Attention (n_heads)
  - Feed Forward Network
  - Layer Normalization
  - Dropout
  (×n_layers)
  ↓
Classification Head: Linear(d_model → vocab_size)
  ↓
출력: (batch_size, seq_len, vocab_size)
```

### 특징 추출 파이프라인

1. **비디오 프레임 처리**
   - OpenCV로 프레임 추출
   - 실시간 프레임 레이트 동기화

2. **손 랜드마크 추출**
   - MediaPipe Hands 모듈
   - 21개 포인트 × 3차원 × 2손 = 126차원

3. **포즈 랜드마크 추출**
   - MediaPipe Pose 모듈  
   - 상체 6개 주요 포인트 × 3차원 = 18차원

4. **시퀀스 생성**
   - 슬라이딩 윈도우 (window_size=32)
   - 50% 오버랩으로 데이터 증강

## 📈 성능 및 평가

### 예상 성능 지표

- **어휘 크기**: 500-1000개 수어 단어
- **시퀀스 정확도**: 85-92% (데이터셋 품질에 따라)
- **실시간 처리**: 30 FPS (Intel GPU)
- **메모리 사용량**: 4-8GB (배치 크기에 따라)

### 평가 메트릭

- **Token-level Accuracy**: 개별 수어 단어 정확도
- **Sequence-level Accuracy**: 전체 문장 정확도  
- **BLEU Score**: 시퀀스 유사도 측정
- **Inference Speed**: 초당 처리 프레임 수

### 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir ./models/logs

# 훈련 로그 확인
tail -f korean_sign_recognition.log
```

## 🔧 문제 해결

### 일반적인 문제

#### 1. Intel GPU 인식 실패

```bash
# Intel GPU 드라이버 확인
intel_gpu_top

# OneAPI 환경 설정
source /opt/intel/oneapi/setvars.sh

# Python에서 확인
python -c "import torch; print('XPU available:', torch.xpu.is_available())"
```

#### 2. 메모리 부족

```bash
# 배치 크기 감소
python main.py train --batch_size 8

# 시퀀스 길이 감소
python main.py train --sequence_length 16

# 모델 크기 감소
python main.py train --d_model 128 --n_layers 4
```

#### 3. MediaPipe 설치 오류

```bash
# 시스템 의존성 설치 (Ubuntu)
sudo apt-get install libgl1-mesa-glx

# macOS
brew install opencv

# MediaPipe 재설치
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

#### 4. CUDA vs Intel GPU 충돌

```bash
# CUDA 비활성화
export CUDA_VISIBLE_DEVICES=""

# Intel GPU 강제 사용
export USE_INTEL_GPU=1
```

### 성능 최적화

#### 1. Intel GPU 최적화

```python
# 코드에서 최적화 활성화
import intel_extension_for_pytorch as ipex

model = ipex.optimize(model)
```

#### 2. 데이터 로딩 최적화

```bash
# 워커 프로세스 수 조정
python main.py train --num_workers 8

# 메모리 핀 활성화는 자동으로 설정됨
```

#### 3. 혼합 정밀도 훈련

```python
# FP16 훈련 (메모리 절약)
from torch.cuda.amp import autocast, GradScaler

# 코드 수정 시 autocast 사용
```

### 로그 및 디버깅

```bash
# 상세 로그 활성화
python main.py train --log_level DEBUG

# 특정 모듈 로그 확인
grep "data_preprocessor" korean_sign_recognition.log

# GPU 메모리 모니터링
watch -n 1 'intel_gpu_top'  # Intel GPU
# 또는
watch -n 1 'nvidia-smi'     # NVIDIA GPU
```

## 📁 출력 파일 구조

```
processed_data/
├── train_data.pt          # 훈련 데이터
├── val_data.pt           # 검증 데이터
└── vocab.json            # 어휘 사전

models/
├── best_model.pt         # 최고 성능 모델
├── final_model.pt        # 최종 모델
├── checkpoint_epoch_*.pt # 중간 체크포인트
├── training_history.json # 훈련 이력
└── logs/                 # TensorBoard 로그
```

## 🤝 기여 방법

1. 이슈 리포트: 버그나 개선사항 제안
2. 코드 기여: Pull Request 환영
3. 데이터셋 기여: 추가 수어 데이터 제공
4. 문서 개선: README나 주석 개선

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

- **이슈 트래커**: GitHub Issues
- **이메일**: [개발자 이메일]
- **위키**: [프로젝트 위키 링크]

---

**참고**: 본 시스템은 연구 및 교육 목적으로 개발되었습니다. 상업적 사용 시 관련 라이선스를 확인하시기 바랍니다.