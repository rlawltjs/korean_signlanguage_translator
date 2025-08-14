#!/bin/bash

# Intel GPU 지원을 위한 설치 스크립트
# 이 스크립트는 선택적으로 실행하세요

echo "한국 수어 인식 시스템 설치 스크립트"
echo "=================================="

# 현재 PyTorch 버전 확인
echo "현재 환경 확인 중..."
python -c "import torch; print(f'Current PyTorch version: {torch.__version__}')" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "설치 옵션을 선택하세요:"
echo "1) 기본 설치 (CPU + CUDA 지원)"
echo "2) Intel GPU 지원 포함 설치"
echo "3) 종료"
read -p "선택 (1-3): " choice

case $choice in
    1)
        echo "기본 설치를 진행합니다..."
        
        # 기본 패키지 설치
        echo "필수 패키지 설치 중..."
        pip install -r requirements.txt
        
        echo ""
        echo "기본 설치 완료!"
        ;;
        
    2)
        echo "Intel GPU 지원 설치를 진행합니다..."
        echo "주의: Intel GPU가 있는 시스템에서만 작동합니다."
        
        # 기본 패키지 먼저 설치
        echo "기본 패키지 설치 중..."
        pip install -r requirements.txt
        
        # Intel Extension for PyTorch 설치 시도
        echo "Intel Extension for PyTorch 설치 중..."
        pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ || {
            echo "Intel Extension 설치 실패. 기본 설치로 진행합니다."
        }
        
        # Intel MKL 설치 시도 (conda 환경에서만)
        if command -v conda &> /dev/null; then
            echo "Intel MKL 설치 중... (성능 향상)"
            conda install -y mkl mkl-include || echo "MKL 설치 실패 (선택사항)"
        fi
        
        echo ""
        echo "Intel GPU 지원 설치 완료!"
        ;;
        
    3)
        echo "설치를 취소합니다."
        exit 0
        ;;
        
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

# 설치 확인
echo ""
echo "설치 확인 중..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'PyTorch import error: {e}')

try:
    import intel_extension_for_pytorch as ipex
    print('Intel Extension for PyTorch: Available')
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f'Intel GPU: Available - {torch.xpu.get_device_name()}')
        else:
            print('Intel GPU: Hardware not detected')
    except:
        print('Intel GPU: Not available')
except ImportError:
    print('Intel Extension for PyTorch: Not installed')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError:
    print('OpenCV: Not available')

try:
    import mediapipe as mp
    print(f'MediaPipe version: {mp.__version__}')
except ImportError:
    print('MediaPipe: Not available')
"

echo ""
echo "설치 완료! 다음 단계:"
echo "1. python test_setup.py 실행하여 전체 테스트"
echo "2. AIHub 데이터를 ./aihub_data 디렉토리에 배치"
echo "3. python main.py pipeline --data_dir ./aihub_data 실행"