import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Intel GPU 지원을 위한 설정
os.environ['USE_INTEL_GPU'] = '1'

class SignLanguagePreprocessor:
    """
    한국 수어 데이터 전처리 클래스
    AIHub 데이터셋을 OpenHands 모델에 맞게 전처리
    """
    
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_json_annotations(self, json_path: str) -> Dict:
        """JSON 어노테이션 파일 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"JSON 로드 실패 {json_path}: {e}")
            return {}
    
    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """손 랜드마크 추출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_points)
        
        # 최대 2개 손까지 지원 (21개 랜드마크 * 3 좌표 * 2 손 = 126개 특징)
        if len(landmarks) == 0:
            return np.zeros(126)  # 손이 감지되지 않은 경우
        elif len(landmarks) == 63:  # 한 손만 감지된 경우
            landmarks.extend([0] * 63)  # 나머지 손은 0으로 패딩
        elif len(landmarks) > 126:  # 2개 손보다 많이 감지된 경우 (에러 방지)
            landmarks = landmarks[:126]
            
        return np.array(landmarks, dtype=np.float32)
    
    def extract_pose_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """포즈 랜드마크 추출 (상체 중심)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # 상체 주요 포인트만 추출 (어깨, 팔꿈치, 손목)
            important_indices = [11, 12, 13, 14, 15, 16]  # 양쪽 어깨, 팔꿈치, 손목
            landmarks = []
            for idx in important_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks, dtype=np.float32)
        else:
            return np.zeros(18)  # 6개 포인트 * 3 좌표 = 18개 특징
    
    def process_video(self, video_path: str, json_data: Dict) -> Optional[Dict]:
        """개별 비디오 처리"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"비디오 열기 실패: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # JSON에서 수어 구간 정보 추출
            sign_segments = json_data.get('data', [])
            
            all_features = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # 수어 구간 내의 프레임만 처리
                is_sign_frame = False
                current_label = "unknown"
                
                for segment in sign_segments:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', duration)
                    
                    if start_time <= timestamp <= end_time:
                        is_sign_frame = True
                        if segment.get('attributes') and len(segment['attributes']) > 0:
                            current_label = segment['attributes'][0].get('name', 'unknown')
                        break
                
                if is_sign_frame:
                    # 손 랜드마크 추출
                    hand_features = self.extract_hand_landmarks(frame)
                    
                    # 포즈 랜드마크 추출
                    pose_features = self.extract_pose_landmarks(frame)
                    
                    # 특징 결합
                    if hand_features is not None and pose_features is not None:
                        combined_features = np.concatenate([hand_features, pose_features])
                        all_features.append({
                            'features': combined_features,
                            'timestamp': timestamp,
                            'label': current_label,
                            'frame_idx': frame_idx
                        })
                
                frame_idx += 1
            
            cap.release()
            
            if len(all_features) == 0:
                self.logger.warning(f"특징점이 추출되지 않음: {video_path}")
                return None
            
            return {
                'video_path': video_path,
                'features': all_features,
                'metadata': json_data.get('metaData', {}),
                'total_frames': len(all_features),
                'labels': list(set([f['label'] for f in all_features]))
            }
            
        except Exception as e:
            self.logger.error(f"비디오 처리 중 오류 {video_path}: {e}")
            return None
    
    def create_sequences(self, features_data: List[Dict], sequence_length: int = 32) -> List[Dict]:
        """시퀀스 데이터 생성"""
        sequences = []
        
        for video_data in features_data:
            features = video_data['features']
            video_path = video_data['video_path']
            
            if len(features) < sequence_length:
                # 짧은 시퀀스는 패딩
                padded_features = features + [features[-1]] * (sequence_length - len(features))
                sequences.append({
                    'sequence': [f['features'] for f in padded_features],
                    'labels': [f['label'] for f in padded_features],
                    'video_path': video_path,
                    'sequence_type': 'padded'
                })
            else:
                # 긴 시퀀스는 슬라이딩 윈도우로 분할
                for i in range(0, len(features) - sequence_length + 1, sequence_length // 2):
                    sequence_features = features[i:i + sequence_length]
                    sequences.append({
                        'sequence': [f['features'] for f in sequence_features],
                        'labels': [f['label'] for f in sequence_features],
                        'video_path': video_path,
                        'sequence_type': 'windowed'
                    })
        
        return sequences
    
    def build_vocabulary(self, sequences: List[Dict]) -> Dict[str, int]:
        """수어 단어 어휘 사전 구축"""
        all_labels = set()
        for seq in sequences:
            all_labels.update(seq['labels'])
        
        # 특수 토큰 추가
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of Sequence
            '<EOS>': 3   # End of Sequence
        }
        
        for i, label in enumerate(sorted(all_labels)):
            if label not in vocab:
                vocab[label] = len(vocab)
        
        return vocab
    
    def save_processed_data(self, sequences: List[Dict], vocab: Dict[str, int], 
                          train_ratio: float = 0.8) -> None:
        """전처리된 데이터 저장"""
        # 훈련/검증 데이터 분할
        np.random.shuffle(sequences)
        split_idx = int(len(sequences) * train_ratio)
        
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # 데이터 저장
        train_data = {
            'sequences': train_sequences,
            'vocab': vocab,
            'num_features': 144  # 126 (손) + 18 (포즈)
        }
        
        val_data = {
            'sequences': val_sequences,
            'vocab': vocab,
            'num_features': 144
        }
        
        # PyTorch 2.6 호환성을 위해 안전한 저장 방식 사용
        torch.save(train_data, self.output_dir / 'train_data.pt', _use_new_zipfile_serialization=False)
        torch.save(val_data, self.output_dir / 'val_data.pt', _use_new_zipfile_serialization=False)
        
        # 어휘 사전 따로 저장
        with open(self.output_dir / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"데이터 저장 완료:")
        self.logger.info(f"  - 훈련 시퀀스: {len(train_sequences)}")
        self.logger.info(f"  - 검증 시퀀스: {len(val_sequences)}")
        self.logger.info(f"  - 어휘 크기: {len(vocab)}")
    
    def process_dataset(self, sequence_length: int = 32, train_ratio: float = 0.8) -> None:
        """전체 데이터셋 처리 파이프라인"""
        self.logger.info("데이터셋 처리 시작...")
        
        # 모든 비디오와 JSON 파일 찾기
        video_files = list(self.data_root.glob('**/*.mp4'))
        self.logger.info(f"발견된 MP4 파일: {len(video_files)}개")
        
        if len(video_files) == 0:
            self.logger.error("MP4 파일을 찾을 수 없습니다.")
            self.logger.error(f"데이터 경로를 확인하세요: {self.data_root}")
            return
        
        # 처음 몇 개 파일명 출력 (디버깅용)
        self.logger.info("파일명 예시:")
        for i, video_path in enumerate(video_files[:3]):
            self.logger.info(f"  {i+1}. {video_path.name}")
        
        processed_data = []
        
        for video_path in video_files:
            # 여러 가지 JSON 파일명 패턴 시도 (순서대로)
            json_candidates = [
                # 패턴 1: 단순 확장자 교체 (.mp4 → .json)
                video_path.with_suffix('.json'),
                
                # 패턴 2: _morpheme.json 추가
                video_path.parent / (video_path.stem + '_morpheme.json'),
                
                # 패턴 3: _annotation.json 추가  
                video_path.parent / (video_path.stem + '_annotation.json'),
                
                # 패턴 4: _label.json 추가
                video_path.parent / (video_path.stem + '_label.json'),
                
                # 패턴 5: .txt → .json 변경 (일부 데이터셋)
                video_path.with_suffix('.txt').with_suffix('.json'),
            ]
            
            # JSON 파일 찾기
            json_path = None
            pattern_used = None
            for i, candidate in enumerate(json_candidates):
                if candidate.exists():
                    json_path = candidate
                    pattern_names = ['.json', '_morpheme.json', '_annotation.json', '_label.json', '.txt→.json']
                    pattern_used = pattern_names[i]
                    break
            
            if json_path is None:
                self.logger.warning(f"JSON 파일 없음: {video_path.name}")
                
                # 디버깅: 해당 디렉토리의 실제 JSON 파일들 확인
                json_files_in_dir = list(video_path.parent.glob('*.json'))
                if json_files_in_dir:
                    self.logger.info(f"  해당 디렉토리의 JSON 파일들:")
                    for json_file in json_files_in_dir[:3]:  # 최대 3개만 표시
                        self.logger.info(f"    - {json_file.name}")
                    
                    # 유사한 파일명 찾기
                    video_stem = video_path.stem
                    similar_files = [f for f in json_files_in_dir if video_stem[:15] in f.name]
                    if similar_files:
                        self.logger.info(f"  유사한 파일명:")
                        for similar_file in similar_files[:2]:
                            self.logger.info(f"    - {similar_file.name}")
                
                continue
            
            # 성공적으로 찾은 경우 패턴 정보 출력 (처음 5개만)
            if len(processed_data) < 5:
                self.logger.info(f"매칭 성공: {video_path.name} → {json_path.name} (패턴: {pattern_used})")
            
            # JSON 데이터 로드
            json_data = self.load_json_annotations(json_path)
            if not json_data:
                continue
            
            # 비디오 처리
            self.logger.info(f"처리 중: {video_path.name}")
            result = self.process_video(str(video_path), json_data)
            
            if result:
                processed_data.append(result)
        
        if not processed_data:
            self.logger.error("처리된 데이터가 없습니다.")
            self.logger.error("파일명 패턴이나 데이터 구조를 확인하세요.")
            return
        
        # 시퀀스 생성
        self.logger.info("시퀀스 생성 중...")
        sequences = self.create_sequences(processed_data, sequence_length)
        
        # 어휘 사전 구축
        self.logger.info("어휘 사전 구축 중...")
        vocab = self.build_vocabulary(sequences)
        
        # 데이터 저장
        self.logger.info("데이터 저장 중...")
        self.save_processed_data(sequences, vocab, train_ratio)
        
        self.logger.info("전처리 완료!")


class SignLanguageDataset(Dataset):
    """PyTorch 데이터셋 클래스"""
    
    def __init__(self, data_path: str, max_length: int = 32):
        # PyTorch 2.6 호환성을 위해 weights_only=False 명시
        self.data = torch.load(data_path, weights_only=False)
        self.sequences = self.data['sequences']
        self.vocab = self.data['vocab']
        self.num_features = self.data['num_features']
        self.max_length = max_length
        
        # 역방향 어휘 사전
        self.idx2word = {v: k for k, v in self.vocab.items()}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        
        # 특징 시퀀스 (T x F) -> (max_length x num_features)
        features = np.array(sequence_data['sequence'])
        labels = sequence_data['labels']
        
        # 패딩 또는 자르기
        if len(features) < self.max_length:
            # 패딩
            pad_length = self.max_length - len(features)
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
            labels = labels + ['<PAD>'] * pad_length
        else:
            # 자르기
            features = features[:self.max_length]
            labels = labels[:self.max_length]
        
        # 레이블을 인덱스로 변환
        label_indices = [self.vocab.get(label, self.vocab['<UNK>']) for label in labels]
        
        return {
            'features': torch.FloatTensor(features),  # (T, F)
            'labels': torch.LongTensor(label_indices),  # (T,)
            'video_path': sequence_data['video_path']
        }


if __name__ == "__main__":
    # 사용 예시
    data_root = "./aihub_sign_data"  # AIHub 데이터 경로
    output_dir = "./processed_data"   # 전처리 결과 저장 경로
    
    preprocessor = SignLanguagePreprocessor(data_root, output_dir)
    preprocessor.process_dataset(sequence_length=32, train_ratio=0.8)
