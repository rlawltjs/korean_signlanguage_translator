#!/usr/bin/env python3
"""
한국 수어 인식 시스템 메인 실행 스크립트
AIHub 데이터를 이용한 OpenHands 모델 파인튜닝
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 로컬 모듈 import
from data_preprocessor import SignLanguagePreprocessor
from openhands_finetuner import finetune_openhands_korean_sign, SignLanguageInference, setup_device


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('korean_sign_recognition.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def preprocess_data(args):
    """데이터 전처리 실행"""
    print("=" * 60)
    print("데이터 전처리 시작")
    print("=" * 60)
    
    # 입력 경로 확인
    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_root}")
    
    # MP4 파일 개수 확인
    mp4_files = list(data_root.glob('**/*.mp4'))
    json_files = list(data_root.glob('**/*_morpheme.json'))
    
    print(f"발견된 MP4 파일: {len(mp4_files)}개")
    print(f"발견된 JSON 파일: {len(json_files)}개")
    
    if len(mp4_files) == 0:
        raise ValueError("MP4 파일이 발견되지 않았습니다.")
    
    # 전처리기 초기화
    preprocessor = SignLanguagePreprocessor(
        data_root=str(data_root),
        output_dir=args.output_dir
    )
    
    # 전처리 실행
    preprocessor.process_dataset(
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio
    )
    
    print("데이터 전처리 완료!")


def train_model(args):
    """모델 훈련 실행"""
    print("=" * 60)
    print("모델 훈련 시작")
    print("=" * 60)
    
    # 전처리된 데이터 확인
    processed_data_dir = Path(args.processed_data_dir)
    train_data_path = processed_data_dir / 'train_data.pt'
    val_data_path = processed_data_dir / 'val_data.pt'
    
    if not train_data_path.exists() or not val_data_path.exists():
        raise FileNotFoundError(
            f"전처리된 데이터를 찾을 수 없습니다. "
            f"먼저 'preprocess' 명령을 실행하세요."
        )
    
    # 디바이스 정보 출력
    device = setup_device()
    print(f"사용 디바이스: {device}")
    
    # 훈련 실행
    history, trainer = finetune_openhands_korean_sign(
        data_dir=args.processed_data_dir,
        save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_length=args.sequence_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    # 훈련 결과 요약
    best_epoch = max(history, key=lambda x: x['val_accuracy'])
    print("\n" + "=" * 60)
    print("훈련 완료!")
    print(f"최고 성능 - 에포크 {best_epoch['epoch']}: "
          f"검증 정확도 {best_epoch['val_accuracy']*100:.2f}%")
    print(f"모델 저장 위치: {args.model_save_dir}")
    print("=" * 60)


def test_inference(args):
    """추론 테스트"""
    print("=" * 60)
    print("추론 테스트")
    print("=" * 60)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 추론기 초기화
    inference = SignLanguageInference(str(model_path))
    
    # 테스트 데이터로 추론 (예시)
    import torch
    from data_preprocessor import SignLanguageDataset
    
    # 검증 데이터셋 로드
    val_dataset = SignLanguageDataset(
        os.path.join(args.processed_data_dir, 'val_data.pt'),
        max_length=32
    )
    
    # 첫 번째 샘플로 테스트
    sample = val_dataset[0]
    features = sample['features'].numpy()
    true_labels = sample['labels'].numpy()
    
    # 예측 실행
    results = inference.predict(features)
    
    print(f"입력 시퀀스 길이: {len(features)}")
    print(f"예측 결과: {results[0]}")
    
    # 실제 레이블과 비교
    true_words = [val_dataset.idx2word[idx] for idx in true_labels if idx != val_dataset.vocab['<PAD>']]
    print(f"실제 레이블: {true_words}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='한국 수어 인식 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 1. 데이터 전처리
  python main.py preprocess --data_dir ./aihub_data --output_dir ./processed_data
  
  # 2. 모델 훈련
  python main.py train --processed_data_dir ./processed_data --model_save_dir ./models
  
  # 3. 추론 테스트
  python main.py inference --model_path ./models/best_model.pt --processed_data_dir ./processed_data
  
  # 4. 전체 파이프라인 실행
  python main.py pipeline --data_dir ./aihub_data --output_dir ./processed_data --model_save_dir ./models
        """
    )
    
    # 서브커맨드 설정
    subparsers = parser.add_subparsers(dest='command', help='실행할 작업')
    
    # 전처리 커맨드
    preprocess_parser = subparsers.add_parser('preprocess', help='데이터 전처리')
    preprocess_parser.add_argument('--data_dir', required=True, help='AIHub 원본 데이터 디렉토리')
    preprocess_parser.add_argument('--output_dir', default='./processed_data', help='전처리 결과 저장 디렉토리')
    preprocess_parser.add_argument('--sequence_length', type=int, default=32, help='시퀀스 길이')
    preprocess_parser.add_argument('--train_ratio', type=float, default=0.8, help='훈련 데이터 비율')
    
    # 훈련 커맨드
    train_parser = subparsers.add_parser('train', help='모델 훈련')
    train_parser.add_argument('--processed_data_dir', default='./processed_data', help='전처리된 데이터 디렉토리')
    train_parser.add_argument('--model_save_dir', default='./models', help='모델 저장 디렉토리')
    train_parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    train_parser.add_argument('--num_epochs', type=int, default=50, help='에포크 수')
    train_parser.add_argument('--sequence_length', type=int, default=32, help='시퀀스 길이')
    train_parser.add_argument('--d_model', type=int, default=256, help='모델 차원')
    train_parser.add_argument('--n_heads', type=int, default=8, help='어텐션 헤드 수')
    train_parser.add_argument('--n_layers', type=int, default=6, help='트랜스포머 레이어 수')
    
    # 추론 커맨드
    inference_parser = subparsers.add_parser('inference', help='추론 테스트')
    inference_parser.add_argument('--model_path', required=True, help='훈련된 모델 경로')
    inference_parser.add_argument('--processed_data_dir', default='./processed_data', help='전처리된 데이터 디렉토리')
    
    # 전체 파이프라인 커맨드
    pipeline_parser = subparsers.add_parser('pipeline', help='전체 파이프라인 실행 (전처리 + 훈련)')
    pipeline_parser.add_argument('--data_dir', required=True, help='AIHub 원본 데이터 디렉토리')
    pipeline_parser.add_argument('--output_dir', default='./processed_data', help='전처리 결과 저장 디렉토리')
    pipeline_parser.add_argument('--model_save_dir', default='./models', help='모델 저장 디렉토리')
    pipeline_parser.add_argument('--sequence_length', type=int, default=32, help='시퀀스 길이')
    pipeline_parser.add_argument('--train_ratio', type=float, default=0.8, help='훈련 데이터 비율')
    pipeline_parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    pipeline_parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    pipeline_parser.add_argument('--num_epochs', type=int, default=50, help='에포크 수')
    pipeline_parser.add_argument('--d_model', type=int, default=256, help='모델 차원')
    pipeline_parser.add_argument('--n_heads', type=int, default=8, help='어텐션 헤드 수')
    pipeline_parser.add_argument('--n_layers', type=int, default=6, help='트랜스포머 레이어 수')
    
    # 공통 인자
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    # 커맨드 실행
    try:
        if args.command == 'preprocess':
            preprocess_data(args)
            
        elif args.command == 'train':
            train_model(args)
            
        elif args.command == 'inference':
            test_inference(args)
            
        elif args.command == 'pipeline':
            # 전처리 실행
            preprocess_args = argparse.Namespace(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                sequence_length=args.sequence_length,
                train_ratio=args.train_ratio
            )
            preprocess_data(preprocess_args)
            
            # 훈련 실행
            train_args = argparse.Namespace(
                processed_data_dir=args.output_dir,
                model_save_dir=args.model_save_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                sequence_length=args.sequence_length,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers
            )
            train_model(train_args)
            
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()