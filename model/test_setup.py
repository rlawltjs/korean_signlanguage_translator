#!/usr/bin/env python3
"""
시스템 설정 테스트 스크립트
설치가 올바르게 되었는지 확인합니다.
"""

import sys
import torch
import numpy as np

def test_pytorch_installation():
    """PyTorch 설치 테스트"""
    print("=== PyTorch 설치 테스트 ===")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 간단한 텐서 연산 테스트
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✅ 기본 텐서 연산 테스트 통과")
    except Exception as e:
        print(f"❌ 기본 텐서 연산 테스트 실패: {e}")
        return False
    
    return True

def test_intel_gpu():
    """Intel GPU 테스트"""
    print("\n=== Intel GPU 테스트 ===")
    try:
        import intel_extension_for_pytorch as ipex
        print("✅ Intel Extension for PyTorch 로드 성공")
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"✅ Intel GPU 사용 가능: {torch.xpu.get_device_name()}")
            
            # Intel GPU 텐서 연산 테스트
            device = torch.device('xpu')
            x = torch.randn(3, 3, device=device)
            y = torch.randn(3, 3, device=device)
            z = torch.mm(x, y)
            print("✅ Intel GPU 텐서 연산 테스트 통과")
            
            return True
        else:
            print("⚠️ Intel GPU 하드웨어가 감지되지 않음")
            return False
            
    except (ImportError, AttributeError, SystemExit) as e:
        print(f"❌ Intel Extension 로드 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Intel GPU 테스트 실패: {e}")
        return False

def test_dependencies():
    """필수 의존성 테스트"""
    print("\n=== 의존성 테스트 ===")
    dependencies = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} 사용 가능")
        except ImportError as e:
            print(f"❌ {name} 없음: {e}")
            all_good = False
    
    return all_good

def test_mediapipe():
    """MediaPipe 기능 테스트"""
    print("\n=== MediaPipe 테스트 ===")
    try:
        import mediapipe as mp
        
        # Hands 모델 초기화 테스트
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        print("✅ MediaPipe Hands 초기화 성공")
        
        # Pose 모델 초기화 테스트
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.7
        )
        print("✅ MediaPipe Pose 초기화 성공")
        
        hands.close()
        pose.close()
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe 테스트 실패: {e}")
        return False

def test_model_components():
    """모델 컴포넌트 테스트"""
    print("\n=== 모델 컴포넌트 테스트 ===")
    try:
        # 로컬 모듈 import 테스트
        from data_preprocessor import SignLanguagePreprocessor
        print("✅ 데이터 전처리기 import 성공")
        
        from openhands_finetuner import OpenHandsKoreanSignModel, setup_device
        print("✅ 모델 클래스 import 성공")
        
        # 디바이스 설정 테스트
        device = setup_device()
        print(f"✅ 디바이스 설정 성공: {device}")
        
        # 더미 모델 생성 테스트
        model = OpenHandsKoreanSignModel(
            input_dim=144,
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_length=16
        )
        model = model.to(device)
        print("✅ 모델 생성 및 디바이스 이동 성공")
        
        # 더미 입력으로 순전파 테스트
        batch_size, seq_len, input_dim = 2, 8, 144
        dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)
        
        with torch.no_grad():
            output = model(dummy_input)
            expected_shape = (batch_size, seq_len, 100)
            if output.shape == expected_shape:
                print(f"✅ 모델 순전파 테스트 통과: {output.shape}")
            else:
                print(f"❌ 모델 출력 형태 오류: {output.shape} != {expected_shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 컴포넌트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("한국 수어 인식 시스템 설정 테스트")
    print("=" * 50)
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(("PyTorch 설치", test_pytorch_installation()))
    test_results.append(("Intel GPU", test_intel_gpu()))
    test_results.append(("필수 의존성", test_dependencies()))
    test_results.append(("MediaPipe", test_mediapipe()))
    test_results.append(("모델 컴포넌트", test_model_components()))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 모든 테스트 통과! 시스템이 올바르게 설정되었습니다.")
        print("\n다음 단계:")
        print("1. AIHub 데이터를 ./aihub_data 디렉토리에 배치")
        print("2. python main.py pipeline --data_dir ./aihub_data 실행")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        print("requirements.txt의 패키지들이 올바르게 설치되었는지 확인하세요.")
        print("Intel GPU를 사용하려면 install_intel_gpu.sh를 실행하세요.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)