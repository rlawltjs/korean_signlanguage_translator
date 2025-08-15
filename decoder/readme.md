# 한국 수어 인식 및 번역 시스템 (수정중)

OpenHands 모델 기반의 한국어 수어 비디오를 자연어 텍스트로 변환하는 AI 시스템입니다.

## 🚀 주요 기능 (테스트중)

- **단계적 학습**: 단어 단위 → 문장 단위 점진적 학습
- **Encoder-Decoder 아키텍처**: OpenHands (수어) + 한국어 LLM (텍스트)
- **실시간 번역**: 웹캠 또는 비디오 파일에서 실시간 수어 번역
- **문맥 인식**: 이전 대화를 고려한 자연스러운 번역
- **다양한 디코딩**: 빔 서치, 문법 제약, 감정 인식

## 📋 요구사항

### 시스템 요구사항
- Python 3.8+
- CUDA 11.0+ (GPU 사용시) 또는 Intel GPU
- RAM 8GB+ 권장
- 저장공간 10GB+ (모델 및 데이터)

### 필수 패키지
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install mediapipe
pip install opencv-python
pip install numpy pandas
pip install tqdm
pip install pathlib
```

### 선택 패키지 (고급 기능)
```bash
# Intel GPU 지원
pip install intel_extension_for_pytorch

# 한국어 NLP
pip install konlpy

# 평가 메트릭
pip install nltk sacrebleu
```

## 📁 프로젝트 구조

```
korean_sign_recognition/
├── data_preprocessor.py          # 데이터 전처리
├── openhands_finetuner.py       # OpenHands 모델 파인튜닝
├── encoder_decoder_model.py     # Encoder-Decoder 모델
├── advanced_sign_to_text.py     # 고급 기능들
├── main.py                      # 메인 실행 스크립트
├── README.md
├── requirements.txt
├── data/
│   ├── aihub_sign_data/         # AIHub 원본 데이터
│   ├── processed_data/          # 전처리된 데이터
│   └── text_annotations.json   # 자연어 문장 어노테이션
├── models/
│   ├── openhands_pretrained/    # 사전훈련된 OpenHands 모델
│   ├── word_level/              # 단어 단위 모델
│   └── sentence_level/          # 문장 단위 모델
└── logs/                        # 훈련 로그
```

## 🔧 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/your-repo/korean-sign-recognition.git
cd korean-sign-recognition
```

### 2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 데이터 준비
AIHub에서 한국 수어 데이터셋을 다운로드하고 `data/aihub_sign_data/` 폴더에 압축 해제:

```
data/aihub_sign_data/
├── video1.mp4
├── video1_morpheme.json
├── video2.mp4
├── video2_morpheme.json
...
```

## 🎯 사용 방법

### 단계 1: 데이터 전처리

AIHub 데이터를 모델 훈련용으로 전처리:

```bash
python main.py preprocess \
    --data_dir ./data/aihub_sign_data \
    --output_dir ./data/processed_data \
    --sequence_length 32 \
    --train_ratio 0.8
```

**주요 옵션:**
- `--data_dir`: AIHub 원본 데이터 경로
- `--output_dir`: 전처리 결과 저장 경로
- `--sequence_length`: 시퀀스 길이 (기본: 32)
- `--train_ratio`: 훈련/검증 분할 비율 (기본: 0.8)

### 단계 2: 단어 단위 모델 훈련

OpenHands 모델을 단어 단위로 사전훈련:

```bash
python main.py train \
    --processed_data_dir ./data/processed_data \
    --model_save_dir ./models/word_level \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50
```

**주요 옵션:**
- `--batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `--learning_rate`: 학습률 (기본: 1e-4)
- `--num_epochs`: 훈련 에포크 수
- `--d_model`: 모델 차원 (기본: 256)
- `--n_heads`: 어텐션 헤드 수 (기본: 8)

### 단계 3: 문장 단위 모델 훈련

자연어 문장 어노테이션 준비:

```python
# text_annotations.json 예시
{
  "video_001": {
    "sentence": "안녕하세요. 만나서 반갑습니다.",
    "keywords": ["안녕", "만나다", "반갑다"]
  },
  "video_002": {
    "sentence": "오늘 날씨가 정말 좋네요.",
    "keywords": ["오늘", "날씨", "좋다"]
  }
}
```

Encoder-Decoder 모델 훈련:

```bash
python -c "
from encoder_decoder_model import train_sign_to_text_model
train_sign_to_text_model(
    sign_data_dir='./data/processed_data',
    text_annotations_path='./data/text_annotations.json',
    pretrained_encoder_path='./models/word_level/best_model.pt',
    save_dir='./models/sentence_level'
)
"
```

### 단계 4: 추론 및 테스트

훈련된 모델로 추론:

```bash
python main.py inference \
    --model_path ./models/sentence_level/best_model.pt \
    --processed_data_dir ./data/processed_data
```

## 💻 코드 사용 예시

### 기본 추론
```python
from encoder_decoder_model import SignToTextModel
import torch

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignToTextModel.load_from_checkpoint('./models/sentence_level/best_model.pt')
model.to(device)
model.eval()

# 수어 특징 입력 (예시)
sign_features = torch.randn(1, 32, 144).to(device)  # (배치, 시퀀스, 특징)

# 텍스트 생성
generated_texts = model.generate_text(sign_features, max_length=50)
print(f"번역 결과: {generated_texts[0]}")
```

### 실시간 대화형 번역기
```python
from advanced_sign_to_text import InteractiveSignTranslator
import numpy as np

# 번역기 초기화
translator = InteractiveSignTranslator('./models/sentence_level/best_model.pt')

# 수어 시퀀스 번역
sign_sequence = np.random.randn(32, 144)  # 실제로는 MediaPipe에서 추출
result = translator.translate_sign_sequence(
    sign_sequence,
    use_beam_search=True,
    use_context=True
)

print(f"번역: {result['translation']}")
print(f"신뢰도: {result['confidence']:.2f}")
print(f"문맥: {result['context']}")
print(f"대안: {result['suggestions']}")
```

### 웹캠 실시간 번역
```python
import cv2
import mediapipe as mp
from data_preprocessor import SignLanguagePreprocessor

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# 웹캠 시작
cap = cv2.VideoCapture(0)
translator = InteractiveSignTranslator('./models/sentence_level/best_model.pt')

sequence_buffer = []
sequence_length = 32

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 손 랜드마크 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # 특징 추출 (실제 구현 필요)
        features = extract_features_from_landmarks(results.multi_hand_landmarks)
        sequence_buffer.append(features)
        
        # 충분한 프레임이 모이면 번역
        if len(sequence_buffer) >= sequence_length:
            sign_features = np.array(sequence_buffer[-sequence_length:])
            result = translator.translate_sign_sequence(sign_features)
            
            # 화면에 번역 결과 표시
            cv2.putText(frame, result['translation'], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"신뢰도: {result['confidence']:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('수어 번역기', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🔍 평가 및 성능 측정

### 번역 품질 평가
```python
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

def evaluate_model(model, test_loader):
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 예측
            pred_texts = model.generate_text(batch['sign_features'])
            predictions.extend(pred_texts)
            
            # 참조 텍스트
            ref_texts = batch['text_string']
            references.extend([[ref.split()] for ref in ref_texts])
    
    # BLEU 스코어 계산
    bleu_score = corpus_bleu(references, [pred.split() for pred in predictions])
    print(f"BLEU Score: {bleu_score:.4f}")
    
    return bleu_score, predictions, references
```

### 실시간 성능 측정
```python
import time

def benchmark_inference_speed(model, test_features, num_runs=100):
    model.eval()
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate_text(test_features)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"평균 추론 시간: {avg_time:.4f}초")
    print(f"초당 프레임 수: {fps:.2f} FPS")
    
    return avg_time, fps
```

## ⚙️ 고급 설정

### GPU 설정
```python
# CUDA GPU 사용
export CUDA_VISIBLE_DEVICES=0

# Intel GPU 사용 (지원되는 경우)
export USE_INTEL_GPU=1
```

### 모델 최적화
```python
# 모델 양자화 (추론 속도 향상)
from torch.quantization import quantize_dynamic

model_quantized = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 배치 처리 최적화
```python
# 동적 배치 크기
def collate_fn(batch):
    # 시퀀스 길이별로 정렬하여 패딩 최소화
    batch.sort(key=lambda x: x['sign_features'].size(0), reverse=True)
    return default_collate(batch)

train_loader = DataLoader(
    dataset, 
    batch_size=16, 
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)
```

## 🐛 문제 해결

### 자주 발생하는 오류

**1. CUDA 메모리 부족**
```bash
# 배치 크기 줄이기
python main.py train --batch_size 8

# 그래디언트 누적 사용
python main.py train --batch_size 4 --gradient_accumulation_steps 4
```

**2. JSON 파일 매칭 실패**
```bash
# 파일명 패턴 확인
ls data/aihub_sign_data/*.json | head -5
ls data/aihub_sign_data/*.mp4 | head -5

# 로그 레벨을 DEBUG로 설정하여 자세한 정보 확인
python main.py preprocess --log_level DEBUG
```

**3. Intel GPU 인식 실패**
```bash
# Intel Extension 설치 확인
pip install intel_extension_for_pytorch

# CPU 폴백 사용
export USE_INTEL_GPU=0
```

### 성능 최적화 팁

1. **데이터 로딩 최적화**: `num_workers=4`, `pin_memory=True` 사용
2. **Mixed Precision**: AMP 사용으로 메모리 절약 및 속도 향상
3. **모델 병렬화**: 큰 모델의 경우 DataParallel 사용
4. **캐시 활용**: 전처리된 특징을 디스크에 캐시

## 🙏 감사의 말

- [AIHub](https://aihub.or.kr/) - 한국 수어 데이터셋 제공
- [OpenHands](https://github.com/AI4Bharat/OpenHands) - 수어 인식 기반 모델
- [MediaPipe](https://mediapipe.dev/) - 손 랜드마크 추출
- [Transformers](https://huggingface.co/transformers/) - 사전훈련된 언어 모델

---

**💡 Tip**: 더 자세한 기술 문서와 예제는 [Wiki 페이지](https://github.com/your-repo/korean-sign-recognition/wiki)를 참조하세요!