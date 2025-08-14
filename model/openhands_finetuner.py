import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Intel GPU 지원 (안전한 import)
INTEL_GPU_AVAILABLE = False
try:
    import intel_extension_for_pytorch as ipex
    INTEL_GPU_AVAILABLE = True
    print("Intel Extension for PyTorch available")
except (ImportError, AttributeError, SystemExit) as e:
    INTEL_GPU_AVAILABLE = False
    print(f"Intel Extension for PyTorch not available: {e}")
    print("Falling back to standard PyTorch")

from data_preprocessor import SignLanguageDataset


class OpenHandsKoreanSignModel(nn.Module):
    """
    한국 수어 인식을 위한 OpenHands 기반 모델
    Transformer 아키텍처 사용
    """
    
    def __init__(self, 
                 input_dim: int = 144,  # 손 랜드마크 (126) + 포즈 랜드마크 (18)
                 vocab_size: int = 1000,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_seq_length: int = 32,
                 dropout: float = 0.1):
        super(OpenHandsKoreanSignModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 입력 특징 임베딩
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 출력 헤드
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """위치 인코딩 생성"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """패딩 마스크 생성"""
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(x.device)
        return mask >= lengths.unsqueeze(1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) 실제 시퀀스 길이
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # 특징 투영
        x = self.feature_projection(x)  # (batch_size, seq_len, d_model)
        
        # 위치 인코딩 추가
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        x = self.dropout(x)
        
        # 패딩 마스크 생성
        if lengths is not None:
            padding_mask = self.create_padding_mask(x, lengths)
        else:
            padding_mask = None
        
        # Transformer 인코더
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 분류
        output = self.classifier(x)  # (batch_size, seq_len, vocab_size)
        
        return output


class SignLanguageTrainer:
    """수어 인식 모델 훈련 클래스"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 vocab: Dict[str, int],
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        
        # Intel GPU 최적화 (안전하게 시도)
        if INTEL_GPU_AVAILABLE and device.type == 'xpu':
            try:
                self.model = ipex.optimize(self.model)
                print("Intel GPU optimization enabled")
            except Exception as e:
                print(f"Intel GPU optimization failed: {e}")
                print("Continuing without Intel optimization")
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # 손실 함수 (패딩 토큰 무시)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
        
        # 로깅
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 역방향 어휘 사전
        self.idx2word = {v: k for k, v in vocab.items()}
        
        # 최고 성능 추적
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
    def calculate_sequence_lengths(self, labels: torch.Tensor) -> torch.Tensor:
        """실제 시퀀스 길이 계산 (패딩 제외)"""
        pad_token_id = self.vocab['<PAD>']
        lengths = []
        
        for seq in labels:
            # 첫 번째 패딩 토큰의 위치를 찾아 길이 계산
            pad_positions = (seq == pad_token_id).nonzero()
            if len(pad_positions) > 0:
                length = pad_positions[0].item()
            else:
                length = len(seq)
            lengths.append(length)
        
        return torch.tensor(lengths, device=labels.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            features = batch['features'].to(self.device)  # (B, T, F)
            labels = batch['labels'].to(self.device)      # (B, T)
            
            # 실제 시퀀스 길이 계산
            lengths = self.calculate_sequence_lengths(labels)
            
            self.optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(features, lengths)  # (B, T, V)
            
            # 손실 계산 (패딩 토큰 제외)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 메트릭 계산
            with torch.no_grad():
                pred_tokens = outputs.argmax(dim=-1)
                mask = labels != self.vocab['<PAD>']
                correct = ((pred_tokens == labels) * mask).sum().item()
                num_tokens = mask.sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_tokens += num_tokens
            
            # 진행률 업데이트
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{correct/num_tokens*100:.2f}%" if num_tokens > 0 else "0%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': avg_acc
        }
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                lengths = self.calculate_sequence_lengths(labels)
                
                outputs = self.model(features, lengths)
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                
                pred_tokens = outputs.argmax(dim=-1)
                mask = labels != self.vocab['<PAD>']
                correct = ((pred_tokens == labels) * mask).sum().item()
                num_tokens = mask.sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_tokens += num_tokens
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_acc
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], save_path: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'vocab': self.vocab,
            'metrics': metrics,
            'model_config': {
                'input_dim': 144,
                'vocab_size': len(self.vocab),
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            }
        }
        # PyTorch 2.6 호환성을 위해 안전한 저장 방식 사용
        torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
        
    def train(self, 
              num_epochs: int,
              save_dir: str,
              save_every: int = 5,
              early_stopping_patience: int = 10) -> List[Dict[str, float]]:
        """전체 훈련 프로세스"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = []
        patience_counter = 0
        
        self.logger.info(f"훈련 시작: {num_epochs} 에포크")
        self.logger.info(f"디바이스: {self.device}")
        self.logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 훈련
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 학습률 스케줄링
            self.scheduler.step(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 메트릭 기록
            epoch_metrics = {
                'epoch': epoch + 1,
                'lr': current_lr,
                'time': time.time() - start_time,
                **train_metrics,
                **val_metrics
            }
            history.append(epoch_metrics)
            
            # 로깅
            self.logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_accuracy']*100:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']*100:.2f}% | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_metrics['time']:.1f}s"
            )
            
            # 최고 모델 저장
            if val_metrics['val_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_accuracy']
                self.save_checkpoint(
                    epoch + 1, 
                    epoch_metrics, 
                    save_path / 'best_model.pt'
                )
                self.logger.info(f"새로운 최고 성능 모델 저장: {self.best_val_acc*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 정기 저장
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    epoch + 1, 
                    epoch_metrics, 
                    save_path / f'checkpoint_epoch_{epoch+1}.pt'
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 최종 모델 저장
        self.save_checkpoint(
            epoch + 1, 
            epoch_metrics, 
            save_path / 'final_model.pt'
        )
        
        # 훈련 이력 저장
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info("훈련 완료!")
        return history


def setup_device():
    """디바이스 설정 (Intel GPU, CUDA, CPU 순으로 시도)"""
    device = None
    device_name = "Unknown"
    
    # Intel GPU 시도
    if INTEL_GPU_AVAILABLE:
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device('xpu')
                device_name = torch.xpu.get_device_name()
                print(f"Intel GPU 사용: {device_name}")
            else:
                print("Intel GPU 사용 불가")
        except Exception as e:
            print(f"Intel GPU 초기화 실패: {e}")
    
    # CUDA GPU 시도
    if device is None and torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name()
        print(f"CUDA GPU 사용: {device_name}")
    
    # CPU 폴백
    if device is None:
        device = torch.device('cpu')
        device_name = "CPU"
        print("CPU 사용")
    
    return device


def load_datasets(data_dir: str, batch_size: int = 16, max_length: int = 32):
    """데이터셋 로드"""
    train_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'train_data.pt'),
        max_length=max_length
    )
    
    val_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'val_data.pt'),
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.vocab


def finetune_openhands_korean_sign(
    data_dir: str = "./processed_data",
    save_dir: str = "./models",
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    max_seq_length: int = 32,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6
):
    """
    OpenHands 모델 파인튜닝 메인 함수
    
    Args:
        data_dir: 전처리된 데이터 디렉토리
        save_dir: 모델 저장 디렉토리  
        batch_size: 배치 크기
        learning_rate: 학습률
        num_epochs: 에포크 수
        max_seq_length: 최대 시퀀스 길이
        d_model: 모델 차원
        n_heads: 어텐션 헤드 수
        n_layers: 트랜스포머 레이어 수
    """
    
    # 디바이스 설정
    device = setup_device()
    
    # 데이터 로드
    print("데이터 로딩 중...")
    train_loader, val_loader, vocab = load_datasets(
        data_dir, batch_size, max_seq_length
    )
    
    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    print(f"어휘 크기: {len(vocab)}")
    
    # 모델 초기화
    print("모델 초기화 중...")
    model = OpenHandsKoreanSignModel(
        input_dim=144,
        vocab_size=len(vocab),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_length=max_seq_length
    )
    
    # 트레이너 초기화
    trainer = SignLanguageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=device,
        lr=learning_rate
    )
    
    # 훈련 시작
    print("훈련 시작...")
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir=save_dir,
        save_every=5,
        early_stopping_patience=10
    )
    
    return history, trainer


class SignLanguageInference:
    """수어 인식 추론 클래스"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        if device is None:
            device = setup_device()
        
        self.device = device
        
        # 모델 로드 (PyTorch 2.6 호환성)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.vocab = checkpoint['vocab']
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        # 모델 초기화
        config = checkpoint['model_config']
        self.model = OpenHandsKoreanSignModel(
            input_dim=config['input_dim'],
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_length=config['max_seq_length']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"모델 로드 완료: {model_path}")
        print(f"어휘 크기: {len(self.vocab)}")
    
    def predict(self, features: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        수어 시퀀스 예측
        
        Args:
            features: (seq_len, feature_dim) 또는 (batch_size, seq_len, feature_dim)
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            예측 결과 리스트
        """
        if features.ndim == 2:
            features = features[np.newaxis, :]  # 배치 차원 추가
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)  # (B, T, V)
            probs = torch.softmax(outputs, dim=-1)
            pred_tokens = outputs.argmax(dim=-1)
            confidences = probs.max(dim=-1)[0]
        
        results = []
        for batch_idx in range(features_tensor.size(0)):
            sequence_result = []
            
            for t in range(features_tensor.size(1)):
                token_id = pred_tokens[batch_idx, t].item()
                confidence = confidences[batch_idx, t].item()
                word = self.idx2word.get(token_id, '<UNK>')
                
                if confidence >= confidence_threshold and word not in ['<PAD>', '<UNK>']:
                    sequence_result.append({
                        'word': word,
                        'confidence': confidence,
                        'timestamp': t
                    })
            
            results.append(sequence_result)
        
        return results


if __name__ == "__main__":
    # 사용 예시
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenHands 한국 수어 모델 파인튜닝')
    parser.add_argument('--data_dir', default='./processed_data', help='전처리된 데이터 디렉토리')
    parser.add_argument('--save_dir', default='./models', help='모델 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=50, help='에포크 수')
    parser.add_argument('--d_model', type=int, default=256, help='모델 차원')
    parser.add_argument('--n_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--n_layers', type=int, default=6, help='트랜스포머 레이어 수')
    
    args = parser.parse_args()
    
    # 파인튜닝 실행
    history, trainer = finetune_openhands_korean_sign(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    print("파인튜닝 완료!")