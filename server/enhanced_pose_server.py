#!/usr/bin/env python3
"""
포즈 추정 서버: 임의 크기 이미지 수신 → YOLOv8n 사람 1명 검출 → 288x384 리사이즈 → RTMW 포즈 추정 → 결과 반환
수어 인식 기능 추가: MediaPipe 기반 수어 인식 모델 지원 (OpenHands Transformer 모델)
"""

import cv2
import numpy as np
import time
import json
import argparse
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import os

# Flask 웹 서버
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import threading
import queue

# MMPose 관련
from mmpose.apis import init_model, inference_topdown
import torch
import torch.nn as nn

# YOLOv8 관련
from ultralytics import YOLO

# MediaPipe 관련 (수어 인식용)
import mediapipe as mp

# Transformer 관련 (OpenHands 모델용)
try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers 라이브러리가 없습니다. pip install transformers 실행하세요.")
    TRANSFORMERS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetector:
    """YOLOv11을 사용한 사람 1명 검출기 (Intel XPU 지원)"""
    
    def __init__(self, device: str = "auto", model_size: str = "n"):
        self.device = self._determine_device(device)
        self.model_size = model_size
        
        # Intel Extension for PyTorch import (XPU 지원)
        try:
            if self.device == "xpu":
                import intel_extension_for_pytorch as ipex
                print(f"🔧 Intel Extension for PyTorch 로딩됨")
        except ImportError:
            if self.device == "xpu":
                print(f"⚠️ Intel Extension for PyTorch 없음 - CPU로 폴백")
                self.device = "cpu"
        
        print(f"🔧 YOLOv11{model_size} 모델 로딩 중... (디바이스: {self.device})")
        start_time = time.time()
        
        try:
            # YOLOv11 모델 로딩 (YOLOv8 대비 더 빠르고 정확함)
            model_name = f'yolo11{model_size}.pt'
            self.yolo_model = YOLO(model_name)
            
            # XPU 또는 다른 디바이스 설정
            if self.device != 'cpu':
                self.yolo_model.to(self.device)
            
            init_time = time.time() - start_time
            print(f"✅ YOLOv11{model_size} 모델 로딩 완료: {init_time:.2f}초")
            print(f"   - 성능: YOLOv8 대비 더 빠른 추론 속도와 높은 정확도")
            print(f"   - 파라미터: YOLOv8 대비 22% 적은 파라미터로 더 높은 mAP")
            
        except Exception as e:
            print(f"❌ YOLOv11{model_size} 모델 로딩 실패: {e}")
            print(f"🔄 YOLOv8n으로 폴백...")
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                if self.device != 'cpu':
                    self.yolo_model.to(self.device)
                init_time = time.time() - start_time
                print(f"✅ YOLOv8n 폴백 모델 로딩 완료: {init_time:.2f}초")
            except Exception as e2:
                print(f"❌ 폴백 모델도 실패: {e2}")
                raise
    
    def _determine_device(self, device: str) -> str:
        """디바이스 자동 결정 (XPU 우선)"""
        if device == "auto":
            # 1. XPU (Intel Arc GPU) 확인
            try:
                if torch.xpu.is_available():
                    print(f"🎮 Intel Arc GPU (XPU) 감지됨")
                    return "xpu"
            except:
                pass
            
            # 2. CUDA 확인  
            if torch.cuda.is_available():
                print(f"🚀 NVIDIA GPU (CUDA) 감지됨")
                return "cuda"
            
            # 3. CPU 폴백
            print(f"💻 CPU 사용")
            return "cpu"
        else:
            # 사용자 지정 디바이스 검증
            if device == "xpu":
                try:
                    if not torch.xpu.is_available():
                        print("⚠️ XPU 미사용 가능 - CPU로 폴백")
                        return "cpu"
                except:
                    print("⚠️ XPU 확인 실패 - CPU로 폴백")
                    return "cpu"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    print("⚠️ CUDA 미사용 가능 - CPU로 폴백")
                    return "cpu"
            
            return device
    
    def detect_best_person(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[List[float]]:
        """이미지에서 가장 신뢰도 높은 사람 1명 검출
        
        Args:
            image: 입력 이미지 (BGR)
            conf_threshold: 신뢰도 임계값
            
        Returns:
            Optional[List[float]]: [x1, y1, x2, y2, confidence] 형태의 바운딩 박스 또는 None
        """
        try:
            # YOLOv8 추론
            results = self.yolo_model(image, verbose=False)
            
            best_person = None
            best_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # 클래스 0은 'person'
                    person_mask = boxes.cls == 0
                    conf_mask = boxes.conf >= conf_threshold
                    valid_mask = person_mask & conf_mask
                    
                    if valid_mask.any():
                        valid_boxes = boxes.xyxy[valid_mask]  # x1, y1, x2, y2
                        valid_confs = boxes.conf[valid_mask]
                        
                        # 가장 신뢰도가 높은 사람 선택
                        for box, conf in zip(valid_boxes, valid_confs):
                            conf_value = float(conf.cpu())
                            if conf_value > best_confidence:
                                best_confidence = conf_value
                                x1, y1, x2, y2 = box.cpu().numpy()
                                best_person = [float(x1), float(y1), float(x2), float(y2), conf_value]
            
            return best_person
            
        except Exception as e:
            logger.error(f"❌ 사람 검출 실패: {e}")
            return None

class RTMWPoseEstimator:
    """RTMW 포즈 추정기"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 device: str = "auto"):
        
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.device = self._determine_device(device)
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(
            *args, **kwargs, weights_only=False
        ) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        print(f"🔧 RTMW 포즈 모델 로딩 중... (디바이스: {self.device})")
        start_time = time.time()
        
        try:
            self.pose_model = init_model(
                config=self.rtmw_config,
                checkpoint=self.rtmw_checkpoint,
                device=self.device
            )
            
            init_time = time.time() - start_time
            print(f"✅ RTMW 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            print(f"❌ {self.device} 포즈 모델 실패: {e}")
            if self.device != 'cpu':
                print(f"🔄 CPU로 폴백...")
                self.device = 'cpu'
                
                self.pose_model = init_model(
                    config=self.rtmw_config,
                    checkpoint=self.rtmw_checkpoint,
                    device='cpu'
                )
                
                init_time = time.time() - start_time
                print(f"✅ CPU 포즈 모델 로딩 완료: {init_time:.2f}초")
            else:
                raise
    
    def _determine_device(self, device: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            # XPU 확인
            try:
                if torch.xpu.is_available():
                    return "xpu"
            except:
                pass
            
            # CUDA 확인
            if torch.cuda.is_available():
                return "cuda"
            
            return "cpu"
        else:
            # 사용자 지정 디바이스 검증
            if device == "xpu":
                try:
                    if not torch.xpu.is_available():
                        print("⚠️ XPU 미사용 가능 - CPU로 폴백")
                        return "cpu"
                except:
                    print("⚠️ XPU 확인 실패 - CPU로 폴백")
                    return "cpu"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    print("⚠️ CUDA 미사용 가능 - CPU로 폴백")
                    return "cpu"
            
            return device
    
    def crop_and_resize_person(self, image: np.ndarray, bbox: List[float], 
                              target_size: Tuple[int, int] = (288, 384)) -> np.ndarray:
        """바운딩 박스를 기반으로 사람 영역을 크롭하고 288x384로 리사이즈
        
        Args:
            image: 원본 이미지
            bbox: [x1, y1, x2, y2, confidence] 바운딩 박스
            target_size: 목표 크기 (width, height)
            
        Returns:
            np.ndarray: 리사이즈된 288x384 이미지
        """
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox[:4]
            
            # 바운딩 박스 좌표 정수화 및 경계 검사
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # 바운딩 박스 유효성 검사
            if x2 <= x1 or y2 <= y1:
                logger.warning("⚠️ 유효하지 않은 바운딩 박스, 전체 이미지 사용")
                crop = image.copy()
            else:
                # 사람 영역 크롭
                crop = image[y1:y2, x1:x2]
            
            # 288x384로 리사이즈 (RTMW 입력 크기)
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            
            return resized
            
        except Exception as e:
            logger.error(f"❌ 크롭 및 리사이즈 실패: {e}")
            # 실패시 전체 이미지를 리사이즈
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    def estimate_pose_on_crop(self, crop_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """크롭된 이미지에서 포즈 추정"""
        try:
            start_time = time.time()
            
            # 크롭 이미지 전체를 바운딩박스로 사용 (288x384)
            h, w = crop_image.shape[:2]
            full_bbox = [0, 0, w, h]
            
            # MMPose 추론
            results = inference_topdown(
                model=self.pose_model,
                img=crop_image,
                bboxes=[full_bbox],
                bbox_format='xyxy'
            )
            
            pose_time = time.time() - start_time
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                return keypoints, scores, pose_time
            else:
                return np.zeros((133, 2)), np.zeros(133), pose_time
                
        except Exception as e:
            logger.error(f"❌ 포즈 추정 실패: {e}")
            return np.zeros((133, 2)), np.zeros(133), 0.0

class OpenHandsTransformerModel(nn.Module):
    """OpenHands 기반 Transformer 수어 인식 모델"""
    
    def __init__(self, 
                 input_size: int = 144,
                 hidden_size: int = 256,  # 512 -> 256으로 변경
                 num_classes: int = 100,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_length: int = 64):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        
        # 입력 임베딩 (feature_projection으로 이름 변경)
        self.feature_projection = nn.Linear(input_size, hidden_size)
        
        # 위치 인코딩
        self.positional_encoding = self._create_positional_encoding(max_seq_length, hidden_size)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,  # 2048 -> 1024
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 분류 헤드 (256 -> 128 -> 14 구조로 변경)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)   # 128 -> num_classes
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """위치 인코딩 생성"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return pe
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            attention_mask: (batch_size, seq_len) - True for valid positions
        
        Returns:
            torch.Tensor: (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 임베딩
        x = self.feature_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # 위치 인코딩 추가
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Transformer Encoder
        if attention_mask is not None:
            # PyTorch Transformer는 True인 위치를 mask(무시)로 처리
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        if attention_mask is not None:
            # 유효한 토큰들만 평균
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            # 전체 시퀀스 평균
            x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # 분류
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits

class SignLanguageRecognizer:
    """수어 인식기 - MediaPipe + OpenHands Transformer 모델 기반"""
    
    def __init__(self, sign_model_path: str = None, sequence_length: int = 32):
        self.sign_model_path = sign_model_path
        self.sequence_length = sequence_length
        self.sign_model = None
        self.device = self._setup_device()
        
        # MediaPipe 초기화
        print(f"🔧 MediaPipe 초기화 중...")
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 수어 모델 로드 시도
        if sign_model_path and os.path.exists(sign_model_path):
            self._load_transformer_model()
        else:
            print(f"⚠️ 수어 모델 경로 없음 또는 파일 없음: {sign_model_path}")
            print(f"   - 특징 추출만 가능합니다.")
        
        print(f"✅ 수어 인식기 초기화 완료")
    
    def _setup_device(self):
        """디바이스 설정"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")
    
    def _load_transformer_model(self):
        """OpenHands Transformer 수어 인식 모델 로드"""
        try:
            print(f"🔧 OpenHands Transformer 수어 인식 모델 로딩 중: {self.sign_model_path}")
            
            # PyTorch 모델 로드
            checkpoint = torch.load(self.sign_model_path, map_location=self.device)
            
            # 체크포인트의 모델 설정 확인
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model_config = {
                    'input_size': config.get('input_dim', 144),
                    'hidden_size': config.get('d_model', 256),
                    'num_classes': config.get('vocab_size', 14),
                    'num_heads': 8,  # 기본값 사용
                    'num_layers': 6,  # 기본값 사용
                    'dropout': 0.1,   # 기본값 사용
                    'max_seq_length': config.get('max_seq_length', 32)
                }
                
                print(f"📊 모델 설정 로드됨:")
                print(f"   - 입력 크기: {model_config['input_size']}")
                print(f"   - Hidden 크기: {model_config['hidden_size']}")
                print(f"   - 클래스 수: {model_config['num_classes']}")
                print(f"   - 시퀀스 길이: {model_config['max_seq_length']}")
            else:
                raise ValueError("체크포인트에 model_config가 없습니다.")

            # OpenHands Transformer 모델 생성
            self.sign_model = OpenHandsTransformerModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_classes=model_config['num_classes'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                max_seq_length=model_config['max_seq_length']
            )
            
            # 가중치 로드
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # 키 이름 매핑
            mapped_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                k = k.replace('input_projection.', 'feature_projection.')
                mapped_state_dict[k] = v
                
            self.sign_model.load_state_dict(mapped_state_dict)
            self.sign_model.to(self.device)
            self.sign_model.eval()
            
            # ⭐ 핵심 수정 부분: 클래스 이름을 직접 정의
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
                print(f"📚 체크포인트에서 클래스 이름 로드됨")
            else:
                # 수동으로 클래스 이름 매핑 정의
                self.class_names = [
                    "<PAD>",           # class_0
                    "<UNK>",           # class_1
                    "<SOS>",           # class_2
                    "<EOS>",           # class_3
                    "감사하다",         # class_4
                    "감사합니다",       # class_5
                    "나",              # class_6
                    "만나다",          # class_7
                    "만나서",          # class_8
                    # "미안합니다",       # class_9
                    "반갑다",          # class_10
                    "안녕하세요",       # class_11
                    "저",              # class_12
                    # "죄송하다"         # class_13
                ]
                print(f"📚 수동으로 정의된 클래스 이름 사용")
                
                print(f"✅ OpenHands Transformer 모델 로딩 완료")
                print(f"   - 모델 타입: Transformer (OpenHands fine-tuned)")
                print(f"   - 클래스 수: {len(self.class_names)}")
                print(f"   - 디바이스: {self.device}")
                print(f"   - 파라미터 수: {sum(p.numel() for p in self.sign_model.parameters())}")
                
                # 클래스 이름 출력 (디버깅용)
                print(f"🏷️ 클래스 매핑:")
                for i, name in enumerate(self.class_names):
                    print(f"   class_{i}: {name}")
            
        except Exception as e:
            print(f"❌ OpenHands Transformer 모델 로드 실패: {e}")
            print(f"   - 체크포인트 형식을 확인하세요.")
            self.sign_model = None
    
    def extract_features_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """이미지에서 MediaPipe 특징 추출"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 손 랜드마크 추출
            hand_results = self.hands.process(rgb_image)
            hand_features = self._extract_hand_features(hand_results)
            
            # 포즈 랜드마크 추출
            pose_results = self.pose.process(rgb_image)
            pose_features = self._extract_pose_features(pose_results)
            
            if hand_features is not None and pose_features is not None:
                combined_features = np.concatenate([hand_features, pose_features])
                return combined_features
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 특징 추출 실패: {e}")
            return None
    
    def _extract_hand_features(self, results) -> Optional[np.ndarray]:
        """손 랜드마크 추출"""
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_points)
        
        # 126차원으로 패딩/자르기 (양손 21*3*2 = 126)
        if len(landmarks) == 0:
            return np.zeros(126, dtype=np.float32)
        elif len(landmarks) == 63:  # 한 손만 검출된 경우
            landmarks.extend([0] * 63)
        elif len(landmarks) > 126:
            landmarks = landmarks[:126]
        
        return np.array(landmarks, dtype=np.float32)
    
    def _extract_pose_features(self, results) -> Optional[np.ndarray]:
        """포즈 랜드마크 추출 (상체 위주)"""
        if results.pose_landmarks:
            # 중요한 상체 포즈 포인트
            important_indices = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
            landmarks = []
            
            for idx in important_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        else:
            return np.zeros(18, dtype=np.float32)  # 6 * 3 = 18
    
    def predict_sign_sequence(self, feature_sequence: List[np.ndarray], confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """특징 시퀀스에서 수어 예측 (Transformer 기반)"""
        if not self.sign_model:
            return {
                'predictions': [],
                'message': 'Sign model not loaded',
                'has_model': False
            }
        
        try:
            # 시퀀스 길이 확인
            if len(feature_sequence) < self.sequence_length:
                return {
                    'predictions': [],
                    'message': f'Need at least {self.sequence_length} frames, got {len(feature_sequence)}',
                    'has_model': True
                }
            
            # 시퀀스를 정확한 길이로 자르기/패딩
            if len(feature_sequence) > self.sequence_length:
                feature_sequence = feature_sequence[-self.sequence_length:]
            elif len(feature_sequence) < self.sequence_length:
                # 패딩 (앞부분을 0으로 채움)
                padding_length = self.sequence_length - len(feature_sequence)
                padding = [np.zeros_like(feature_sequence[0]) for _ in range(padding_length)]
                feature_sequence = padding + feature_sequence
            
            # 어텐션 마스크 생성 (실제 데이터가 있는 부분만 True)
            attention_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
            actual_length = len([f for f in feature_sequence if not np.all(f == 0)])
            if actual_length > 0:
                attention_mask[-actual_length:] = True
            
            # 텐서로 변환
            features = np.array(feature_sequence)
            features = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, seq_len, feature_dim)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)  # (1, seq_len)
            
            # 예측 수행
            with torch.no_grad():
                outputs = self.sign_model(features, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.topk(probabilities, k=3)  # top 3 predictions
            
            # 결과 정리
            predictions = []
            for conf, idx in zip(confidences[0], predicted_indices[0]):
                conf_val = float(conf.cpu())
                idx_val = int(idx.cpu())
                
                if idx_val < len(self.class_names):
                    class_name = self.class_names[idx_val]
                    if class_name not in {"<PAD>", "<UNK>", "< SOS >", "<EOS>"} and conf_val >= confidence_threshold:
                        predictions.append({
                            'class': class_name,  # 'word'를 'class'로 변경
                            'confidence': conf_val
                        })
        
            if len(predictions) > 0:
                print("\n🤟 수어 인식 결과:")
                print("-" * 40)
                for pred in predictions:
                    print(f"단어: {pred['class']:<15} - 신뢰도: {pred['confidence']*100:>6.2f}%")
                print("-" * 40)
            
            return {
                'predictions': predictions,
                'message': 'Transformer prediction completed',
                'has_model': True,
                'model_type': 'OpenHands Transformer'
            }
            
        except Exception as e:
            print(f"❌ 수어 인식 실패: {e}")
            return {
                'predictions': [],
                'error': str(e),
                'has_model': True,
                'model_type': 'OpenHands Transformer'
            }

class PoseServer:
    """포즈 추정 서버 (수어 인식 기능 추가)"""
    
    def __init__(self, 
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 device: str = "auto",
                 port: int = 5000,
                 host: str = "0.0.0.0",
                 detection_conf: float = 0.5,
                 yolo_model_size: str = "n",
                 sign_model_path: str = None):
        
        self.port = port
        self.host = host
        self.detection_conf = detection_conf
        self.yolo_model_size = yolo_model_size
        
        # YOLOv11 검출기 초기화 (1명만 검출)
        self.detector = PersonDetector(device, yolo_model_size)
        
        # RTMW 추정기 초기화
        self.estimator = RTMWPoseEstimator(rtmw_config, rtmw_checkpoint, device)
        
        # 수어 인식기 초기화
        self.sign_recognizer = SignLanguageRecognizer(sign_model_path)
        
        # Flask 앱 설정
        self.app = Flask(__name__)
        self.setup_routes()
        
        # 성능 통계
        self.request_count = 0
        self.processing_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.sign_request_count = 0
        self.sign_processing_times = deque(maxlen=100)
        
        # 수어 인식용 특징 버퍼 (클라이언트별로 관리)
        self.feature_buffers = {}  # {client_id: deque}
        
        print(f"✅ 포즈 서버 초기화 완료")
        print(f"   - YOLO 모델: v11{yolo_model_size} (성능: v8 대비 빠르고 정확)")
        print(f"   - 검출 디바이스: {self.detector.device}")
        print(f"   - 포즈 디바이스: {self.estimator.device}")
        print(f"   - 검출 신뢰도: {detection_conf}")
        print(f"   - 서버 주소: {host}:{port}")
        print(f"   - 검출 모드: 1명만 검출 (최고 신뢰도)")
        print(f"   - 수어 인식: {'OpenHands Transformer 모델 활성화됨' if self.sign_recognizer.sign_model else '특징추출만 가능'}")

    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """헬스 체크"""
            return jsonify({
                'status': 'healthy',
                'detection_device': self.detector.device,
                'pose_device': self.estimator.device,
                'request_count': self.request_count,
                'sign_request_count': self.sign_request_count,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
                'avg_sign_processing_time': np.mean(self.sign_processing_times) if self.sign_processing_times else 0,
                'detection_mode': 'single_person_best_confidence',
                'features': ['pose_estimation', 'sign_recognition'],
                'sign_model_loaded': self.sign_recognizer.sign_model is not None,
                'sign_model_type': 'OpenHands Transformer' if self.sign_recognizer.sign_model else None
            })
        
        @self.app.route('/estimate_pose', methods=['POST'])
        def estimate_pose():
            """기존 포즈 추정 엔드포인트 (변경 없음)"""
            try:
                start_time = time.time()
                
                # 요청 데이터 검증
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'Empty image file'}), 400
                
                # 메타데이터 파싱
                frame_id = request.form.get('frame_id', 0)
                timestamp = float(request.form.get('timestamp', time.time()))
                detection_only = request.form.get('detection_only', 'false').lower() == 'true'
                
                # 이미지 디코딩
                image_data = image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if original_image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                orig_h, orig_w = original_image.shape[:2]
                
                # 1. 사람 검출 (1명만, 가장 높은 신뢰도)
                detection_start = time.time()
                best_person = self.detector.detect_best_person(original_image, self.detection_conf)
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                if best_person is None:
                    return jsonify({
                        'frame_id': frame_id,
                        'error': 'No person detected',
                        'detection_time': detection_time,
                        'total_time': time.time() - start_time,
                        'timestamp': timestamp,
                        'received_at': start_time,
                        'original_size': [orig_w, orig_h],
                        'detected_persons': 0,
                        'detection_mode': 'single_person_best_confidence'
                    })
                
                # 2. 크롭 및 리사이즈 (288x384)
                crop_image = self.estimator.crop_and_resize_person(original_image, best_person)
                
                # 3. 포즈 추정
                keypoints, scores, pose_time = self.estimator.estimate_pose_on_crop(crop_image)
                
                # 4. 키포인트 좌표 처리
                # 크롭된 영역의 정보
                x1, y1, x2, y2 = best_person[:4]
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                # 288x384 기준 키포인트 (수어 인식 모델용) - 원본 그대로 유지
                keypoints_288x384 = keypoints.copy()
                
                # 원본 이미지 좌표로 변환된 키포인트 (스켈레톤 시각화용)
                keypoints_original = keypoints.copy()
                if keypoints_original.size > 0:
                    # 288x384에서 크롭 영역으로 스케일링
                    keypoints_original[:, 0] = keypoints_original[:, 0] * (crop_w / 288.0) + x1
                    keypoints_original[:, 1] = keypoints_original[:, 1] * (crop_h / 384.0) + y1
                
                # 응답 데이터 구성
                total_time = time.time() - start_time
                self.processing_times.append(total_time)
                self.request_count += 1
                
                response = {
                    'frame_id': frame_id,
                    'keypoints': keypoints_original.tolist(),  # 원본 이미지 좌표 (스켈레톤 시각화용)
                    'keypoints_288x384': keypoints_288x384.tolist(),  # 288x384 좌표 (수어 인식 모델용)
                    'scores': scores.tolist(),
                    'person_box': best_person,
                    'detection_time': detection_time,
                    'pose_time': pose_time,
                    'total_time': total_time,
                    'timestamp': timestamp,
                    'received_at': start_time,
                    'yolo_model': f'v11{self.yolo_model_size}',
                    'detection_device': self.detector.device,
                    'pose_device': self.estimator.device,
                    'original_size': [orig_w, orig_h],
                    'crop_size': [288, 384],
                    'detected_persons': 1,
                    'detection_mode': 'single_person_best_confidence'
                }
                
                # 주기적 로그 출력
                if self.request_count % 30 == 0:
                    avg_time = np.mean(self.processing_times) if self.processing_times else 0
                    avg_det_time = np.mean(self.detection_times) if self.detection_times else 0
                    logger.info(f"📊 요청 {self.request_count}: 평균 처리시간 {avg_time*1000:.1f}ms (검출: {avg_det_time*1000:.1f}ms)")
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ 포즈 추정 요청 실패: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/extract_sign_features', methods=['POST'])
        def extract_sign_features():
            """수어 특징 추출 엔드포인트 (MediaPipe 기반)"""
            try:
                start_time = time.time()
                
                # 요청 데이터 검증
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'Empty image file'}), 400
                
                # 메타데이터 파싱
                frame_id = request.form.get('frame_id', 0)
                timestamp = float(request.form.get('timestamp', time.time()))
                client_id = request.form.get('client_id', 'default')
                
                # 이미지 디코딩
                image_data = image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if original_image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                orig_h, orig_w = original_image.shape[:2]
                
                # MediaPipe로 특징 추출
                extraction_start = time.time()
                features = self.sign_recognizer.extract_features_from_image(original_image)
                extraction_time = time.time() - extraction_start
                
                if features is None:
                    return jsonify({
                        'frame_id': frame_id,
                        'error': 'No hand or pose landmarks detected',
                        'extraction_time': extraction_time,
                        'total_time': time.time() - start_time,
                        'timestamp': timestamp,
                        'received_at': start_time,
                        'original_size': [orig_w, orig_h],
                        'method': 'mediapipe'
                    }), 200
                
                # 클라이언트별 특징 버퍼 관리
                if client_id not in self.feature_buffers:
                    self.feature_buffers[client_id] = deque(maxlen=self.sign_recognizer.sequence_length)
                
                self.feature_buffers[client_id].append(features)
                
                # 응답 데이터 구성
                total_time = time.time() - start_time
                self.sign_processing_times.append(total_time)
                self.sign_request_count += 1
                
                response = {
                    'frame_id': frame_id,
                    'features': features.tolist(),
                    'features_shape': features.shape,
                    'buffer_length': len(self.feature_buffers[client_id]),
                    'buffer_max_length': self.sign_recognizer.sequence_length,
                    'extraction_time': extraction_time,
                    'total_time': total_time,
                    'timestamp': timestamp,
                    'received_at': start_time,
                    'original_size': [orig_w, orig_h],
                    'method': 'mediapipe',
                    'client_id': client_id
                }
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ 수어 특징 추출 실패: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict_sign', methods=['POST'])
        def predict_sign():
            """수어 예측 엔드포인트"""
            try:
                start_time = time.time()
                
                # 메타데이터 파싱
                frame_id = request.form.get('frame_id', 0)
                timestamp = float(request.form.get('timestamp', time.time()))
                client_id = request.form.get('client_id', 'default')
                confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
                
                # 클라이언트 버퍼 확인
                if client_id not in self.feature_buffers:
                    return jsonify({
                        'frame_id': frame_id,
                        'error': 'No feature buffer found for client. Call extract_sign_features first.',
                        'client_id': client_id,
                        'timestamp': timestamp,
                        'total_time': time.time() - start_time
                    }), 400
                
                feature_buffer = self.feature_buffers[client_id]
                
                if len(feature_buffer) == 0:
                    return jsonify({
                        'frame_id': frame_id,
                        'error': 'Feature buffer is empty',
                        'client_id': client_id,
                        'timestamp': timestamp,
                        'total_time': time.time() - start_time
                    }), 400
                
                # 수어 예측 수행
                prediction_start = time.time()
                feature_sequence = list(feature_buffer)
                sign_results = self.sign_recognizer.predict_sign_sequence(
                    feature_sequence, confidence_threshold
                )
                prediction_time = time.time() - prediction_start
                
                # 응답 데이터 구성
                total_time = time.time() - start_time
                
                response = {
                    'frame_id': frame_id,
                    'client_id': client_id,
                    'sign_predictions': sign_results.get('predictions', []),
                    'buffer_length': len(feature_buffer),
                    'sequence_length_required': self.sign_recognizer.sequence_length,
                    'prediction_time': prediction_time,
                    'total_time': total_time,
                    'timestamp': timestamp,
                    'received_at': start_time,
                    'confidence_threshold': confidence_threshold,
                    'message': sign_results.get('message', ''),
                    'error': sign_results.get('error', None),
                    'has_model': sign_results.get('has_model', False),
                    'model_type': sign_results.get('model_type', 'OpenHands Transformer')
                }
                
                # 주기적 로그 출력
                if len(sign_results.get('predictions', [])) > 0:
                    best_pred = sign_results['predictions'][0]
                    logger.info(f"🤟 수어 예측: {best_pred['word']} (신뢰도: {best_pred['confidence']:.2f})")
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ 수어 예측 실패: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/sign_recognition', methods=['POST'])
        def sign_recognition():
            """통합 수어 인식 엔드포인트 (특징 추출 + 예측)"""
            try:
                start_time = time.time()
                
                # 요청 데이터 검증
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'Empty image file'}), 400
                
                # 메타데이터 파싱
                frame_id = request.form.get('frame_id', 0)
                timestamp = float(request.form.get('timestamp', time.time()))
                client_id = request.form.get('client_id', 'default')
                confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
                extract_only = request.form.get('extract_only', 'false').lower() == 'true'
                
                # 이미지 디코딩
                image_data = image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if original_image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                orig_h, orig_w = original_image.shape[:2]
                
                # 1. 특징 추출
                extraction_start = time.time()
                features = self.sign_recognizer.extract_features_from_image(original_image)
                extraction_time = time.time() - extraction_start
                
                if features is None:
                    return jsonify({
                        'frame_id': frame_id,
                        'error': 'No hand or pose landmarks detected',
                        'extraction_time': extraction_time,
                        'total_time': time.time() - start_time,
                        'timestamp': timestamp,
                        'received_at': start_time,
                        'original_size': [orig_w, orig_h],
                        'method': 'integrated',
                        'client_id': client_id
                    }), 200
                
                # 클라이언트별 특징 버퍼 관리
                if client_id not in self.feature_buffers:
                    self.feature_buffers[client_id] = deque(maxlen=self.sign_recognizer.sequence_length)
                
                self.feature_buffers[client_id].append(features)
                
                # 특징 추출만 요청한 경우
                if extract_only:
                    response = {
                        'frame_id': frame_id,
                        'client_id': client_id,
                        'features': features.tolist(),
                        'buffer_length': len(self.feature_buffers[client_id]),
                        'extraction_time': extraction_time,
                        'total_time': time.time() - start_time,
                        'timestamp': timestamp,
                        'received_at': start_time,
                        'original_size': [orig_w, orig_h],
                        'method': 'integrated'
                    }
                    return jsonify(response)
                
                # 2. 수어 예측 수행
                prediction_start = time.time()
                feature_sequence = list(self.feature_buffers[client_id])
                sign_results = self.sign_recognizer.predict_sign_sequence(
                    feature_sequence, confidence_threshold
                )
                prediction_time = time.time() - prediction_start
                
                # 응답 데이터 구성
                total_time = time.time() - start_time
                self.sign_processing_times.append(total_time)
                self.sign_request_count += 1
                
                response = {
                    'frame_id': frame_id,
                    'client_id': client_id,
                    'features': features.tolist(),
                    'sign_predictions': sign_results.get('predictions', []),
                    'buffer_length': len(self.feature_buffers[client_id]),
                    'sequence_length_required': self.sign_recognizer.sequence_length,
                    'extraction_time': extraction_time,
                    'prediction_time': prediction_time,
                    'total_time': total_time,
                    'timestamp': timestamp,
                    'received_at': start_time,
                    'original_size': [orig_w, orig_h],
                    'confidence_threshold': confidence_threshold,
                    'method': 'integrated',
                    'message': sign_results.get('message', ''),
                    'error': sign_results.get('error', None),
                    'has_model': sign_results.get('has_model', False),
                    'model_type': sign_results.get('model_type', 'OpenHands Transformer')
                }
                
                # 주기적 로그 출력
                if self.sign_request_count % 10 == 0:
                    avg_time = np.mean(self.sign_processing_times) if self.sign_processing_times else 0
                    logger.info(f"🤟 수어 인식 요청 {self.sign_request_count}: 평균 처리시간 {avg_time*1000:.1f}ms")
                
                if len(sign_results.get('predictions', [])) > 0:
                    best_pred = sign_results['predictions'][0]
                    logger.info(f"🎯 인식된 수어: {best_pred['class']} (신뢰도: {best_pred['confidence']:.2f})")
                
                print(f"\n🤟 통합 수어 인식 결과: {jsonify(response)}")
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ 수어 인식 실패: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/clear_buffer/<client_id>', methods=['POST'])
        def clear_buffer(client_id):
            """클라이언트별 특징 버퍼 클리어"""
            if client_id in self.feature_buffers:
                self.feature_buffers[client_id].clear()
                return jsonify({
                    'message': f'Buffer cleared for client {client_id}',
                    'client_id': client_id
                })
            else:
                return jsonify({
                    'message': f'No buffer found for client {client_id}',
                    'client_id': client_id
                }), 404
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """통계 정보"""
            buffer_stats = {}
            for client_id, buffer in self.feature_buffers.items():
                buffer_stats[client_id] = len(buffer)
            
            return jsonify({
                'request_count': self.request_count,
                'sign_request_count': self.sign_request_count,
                'detection_device': self.detector.device,
                'pose_device': self.estimator.device,
                'detection_conf': self.detection_conf,
                'detection_mode': 'single_person_best_confidence',
                'sign_model_loaded': self.sign_recognizer.sign_model is not None,
                'sign_model_type': 'OpenHands Transformer' if self.sign_recognizer.sign_model else None,
                'sign_sequence_length': self.sign_recognizer.sequence_length,
                'active_clients': len(self.feature_buffers),
                'client_buffer_stats': buffer_stats,
                'processing_times': {
                    'count': len(self.processing_times),
                    'mean': float(np.mean(self.processing_times)) if self.processing_times else 0,
                    'min': float(np.min(self.processing_times)) if self.processing_times else 0,
                    'max': float(np.max(self.processing_times)) if self.processing_times else 0,
                    'std': float(np.std(self.processing_times)) if self.processing_times else 0
                },
                'detection_times': {
                    'count': len(self.detection_times),
                    'mean': float(np.mean(self.detection_times)) if self.detection_times else 0,
                    'min': float(np.min(self.detection_times)) if self.detection_times else 0,
                    'max': float(np.max(self.detection_times)) if self.detection_times else 0,
                    'std': float(np.std(self.detection_times)) if self.detection_times else 0
                },
                'sign_processing_times': {
                    'count': len(self.sign_processing_times),
                    'mean': float(np.mean(self.sign_processing_times)) if self.sign_processing_times else 0,
                    'min': float(np.min(self.sign_processing_times)) if self.sign_processing_times else 0,
                    'max': float(np.max(self.sign_processing_times)) if self.sign_processing_times else 0,
                    'std': float(np.std(self.sign_processing_times)) if self.sign_processing_times else 0
                },
                'features': ['pose_estimation', 'sign_recognition', 'feature_extraction']
            })
    
    def run(self):
        """서버 실행"""
        print(f"\n🚀 포즈 추정 서버 시작")
        print(f"   - 주소: http://{self.host}:{self.port}")
        print(f"   - 헬스체크: http://{self.host}:{self.port}/health")
        print(f"   - 통계: http://{self.host}:{self.port}/stats")
        print(f"   - 포즈 추정: POST http://{self.host}:{self.port}/estimate_pose")
        print(f"   - 수어 특징 추출: POST http://{self.host}:{self.port}/extract_sign_features")
        print(f"   - 수어 예측: POST http://{self.host}:{self.port}/predict_sign")
        print(f"   - 통합 수어 인식: POST http://{self.host}:{self.port}/sign_recognition")
        print(f"   - 버퍼 클리어: POST http://{self.host}:{self.port}/clear_buffer/<client_id>")
        print(f"   - 검출 모드: 1명만 검출 (최고 신뢰도)")
        print(f"   - 수어 모델: {'OpenHands Transformer' if self.sign_recognizer.sign_model else '특징추출만'}")
        print(f"   - Ctrl+C로 종료")
        
        try:
            # Flask 앱 실행 (디버그 모드 비활성화, 프로덕션용)
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,  # 멀티스레드 지원
                use_reloader=False
            )
        except KeyboardInterrupt:
            print("\n⏹️ 포즈 서버 종료")
        except Exception as e:
            logger.error(f"❌ 서버 실행 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced RTMW Pose Estimation Server with OpenHands Transformer Sign Language Recognition")
    parser.add_argument("--config", type=str, 
                       default="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py",
                       help="RTMW 설정 파일 경로")
    parser.add_argument("--checkpoint", type=str,
                       default="models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
                       help="RTMW 체크포인트 파일 경로") 
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스 (기본값: auto - XPU > CUDA > CPU 순서)")
    parser.add_argument("--yolo-model", type=str, default="n",
                       choices=["n", "s", "m", "l", "x"],
                       help="YOLO 모델 크기 (기본값: n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--port", type=int, default=5000,
                       help="서버 포트 (기본값: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="서버 호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--detection-conf", type=float, default=0.5,
                       help="사람 검출 신뢰도 임계값 (기본값: 0.5)")
    parser.add_argument("--sign-model", type=str, default=None,
                       help="OpenHands fine-tuned Transformer 수어 인식 모델 경로 (예: best_model.pt)")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.config).exists():
        print(f"❌ 설정 파일 없음: {args.config}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"❌ 체크포인트 파일 없음: {args.checkpoint}")
        return
    
    # 수어 모델 파일 확인 (옵션)
    if args.sign_model and not Path(args.sign_model).exists():
        print(f"⚠️ 수어 모델 파일 없음: {args.sign_model} (특징 추출만 가능)")
        args.sign_model = None
    
    # YOLO 모델 성능 정보 출력
    model_info = {
        "n": "nano - 가장 빠름, 가벼움 (3.3M 파라미터)",
        "s": "small - 속도와 정확도 균형 (11.2M 파라미터)", 
        "m": "medium - 더 높은 정확도 (20.1M 파라미터)",
        "l": "large - 높은 정확도 (25.3M 파라미터)",
        "x": "xlarge - 최고 정확도, 느림 (68.2M 파라미터)"
    }
    
    print(f"\n🚀 Enhanced Pose Server 정보:")
    print(f"   - YOLOv11{args.yolo_model} ({model_info[args.yolo_model]})")
    print(f"   - RTMW 포즈 추정: 활성화")
    print(f"   - 수어 인식: {'OpenHands Transformer 모델 로드됨' if args.sign_model else 'MediaPipe 특징추출만'}")
    print(f"   - YOLOv8 대비: 22% 적은 파라미터로 더 높은 mAP")
    print(f"   - 추론 속도: YOLOv8 대비 평균 40% 향상")
    
    if args.device == "auto":
        print(f"\n🔧 디바이스 자동 선택 순서:")
        print(f"   1. XPU (Intel Arc GPU) - 가장 권장")
        print(f"   2. CUDA (NVIDIA GPU)")  
        print(f"   3. CPU (폴백)")
    
    # Transformers 라이브러리 확인
    if args.sign_model and not TRANSFORMERS_AVAILABLE:
        print(f"\n⚠️ transformers 라이브러리 필요:")
        print(f"   pip install transformers")
        print(f"   현재는 특징 추출만 가능합니다.")
        args.sign_model = None
    
    try:
        pose_server = PoseServer(
            rtmw_config=args.config,
            rtmw_checkpoint=args.checkpoint,
            device=args.device,
            port=args.port,
            host=args.host,
            detection_conf=args.detection_conf,
            yolo_model_size=args.yolo_model,
            sign_model_path=args.sign_model
        )
        
        pose_server.run()
        
    except Exception as e:
        print(f"❌ 포즈 서버 실행 실패: {e}")

if __name__ == "__main__":
    main()