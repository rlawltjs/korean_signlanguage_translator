import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertTokenizer, BertModel
)
from pathlib import Path
import json
import time
from tqdm import tqdm

from openhands_finetuner import OpenHandsKoreanSignModel, setup_device
from data_preprocessor import SignLanguageDataset


class SignToTextModel(nn.Module):
    """
    수어 비디오를 자연스러운 텍스트로 변환하는 Encoder-Decoder 모델
    - Encoder: OpenHands 기반 수어 인식 모델
    - Decoder: 사전훈련된 한국어 LLM (GPT-2, T5 등)
    """
    
    def __init__(self, 
                 sign_encoder: OpenHandsKoreanSignModel,
                 text_model_name: str = "skt/kogpt2-base-v2",  # 한국어 GPT-2
                 freeze_encoder: bool = False,
                 cross_attention_dim: int = 256):
        super(SignToTextModel, self).__init__()
        
        self.sign_encoder = sign_encoder
        self.encoder_dim = sign_encoder.d_model
        
        # 텍스트 토크나이저와 모델 로드
        if "kogpt2" in text_model_name or "gpt2" in text_model_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
            self.text_decoder = GPT2LMHeadModel.from_pretrained(text_model_name)
            self.model_type = "gpt2"
        elif "t5" in text_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
            self.text_decoder = T5ForConditionalGeneration.from_pretrained(text_model_name)
            self.model_type = "t5"
        else:
            raise ValueError(f"지원하지 않는 모델: {text_model_name}")
        
        # 특수 토큰 추가
        special_tokens = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Cross-attention 레이어 (수어 표현과 텍스트를 연결)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 프로젝션 레이어
        self.encoder_projection = nn.Linear(self.encoder_dim, cross_attention_dim)
        self.decoder_projection = nn.Linear(
            self.text_decoder.config.hidden_size, cross_attention_dim
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cross_attention_dim, self.text_decoder.config.hidden_size)
        )
        
        # 인코더 동결 옵션
        if freeze_encoder:
            for param in self.sign_encoder.parameters():
                param.requires_grad = False
    
    def encode_sign_sequence(self, sign_features: torch.Tensor, 
                           sign_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """수어 시퀀스 인코딩"""
        # OpenHands 모델로 수어 특징 추출
        sign_outputs = self.sign_encoder(sign_features, sign_lengths)  # (B, T, vocab_size)
        
        # 수어 임베딩으로 변환 (softmax 적용하여 확률 분포를 임베딩으로)
        sign_probs = torch.softmax(sign_outputs, dim=-1)  # (B, T, vocab_size)
        
        # 가중 평균으로 연속적인 표현 생성
        sign_embeddings = torch.matmul(sign_probs, 
                                     self.sign_encoder.classifier[0].weight)  # (B, T, d_model)
        
        return self.encoder_projection(sign_embeddings)  # (B, T, cross_attention_dim)
    
    def forward(self, 
                sign_features: torch.Tensor,
                target_texts: Optional[torch.Tensor] = None,
                sign_lengths: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            sign_features: (B, T_sign, feature_dim) 수어 특징
            target_texts: (B, T_text) 타겟 텍스트 토큰 (훈련시에만)
            sign_lengths: (B,) 수어 시퀀스 길이
            text_lengths: (B,) 텍스트 시퀀스 길이
        """
        batch_size = sign_features.size(0)
        
        # 1. 수어 인코딩
        encoded_signs = self.encode_sign_sequence(sign_features, sign_lengths)  # (B, T_sign, dim)
        
        if target_texts is not None:  # 훈련 모드
            # 2. 텍스트 임베딩
            if self.model_type == "gpt2":
                text_embeddings = self.text_decoder.transformer.wte(target_texts)  # (B, T_text, dim)
            else:  # T5
                text_embeddings = self.text_decoder.shared(target_texts)
            
            text_projected = self.decoder_projection(text_embeddings)  # (B, T_text, cross_dim)
            
            # 3. Cross-attention (텍스트가 수어에 attention)
            attended_features, attention_weights = self.cross_attention(
                query=text_projected,
                key=encoded_signs, 
                value=encoded_signs
            )  # (B, T_text, cross_dim)
            
            # 4. 특징 융합
            fused_features = torch.cat([text_projected, attended_features], dim=-1)  # (B, T_text, cross_dim*2)
            enhanced_text_features = self.fusion_layer(fused_features)  # (B, T_text, text_dim)
            
            # 5. 텍스트 생성
            if self.model_type == "gpt2":
                # GPT-2의 경우 hidden states에 직접 추가
                outputs = self.text_decoder(
                    inputs_embeds=text_embeddings + enhanced_text_features,
                    labels=target_texts
                )
            else:  # T5
                outputs = self.text_decoder(
                    inputs_embeds=text_embeddings + enhanced_text_features,
                    labels=target_texts
                )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'attention_weights': attention_weights,
                'encoded_signs': encoded_signs
            }
        
        else:  # 추론 모드
            return {
                'encoded_signs': encoded_signs
            }
    
    def generate_text(self, 
                     sign_features: torch.Tensor,
                     sign_lengths: Optional[torch.Tensor] = None,
                     max_length: int = 50,
                     temperature: float = 0.8,
                     top_p: float = 0.9) -> List[str]:
        """수어에서 텍스트 생성 (추론)"""
        self.eval()
        batch_size = sign_features.size(0)
        device = sign_features.device
        
        with torch.no_grad():
            # 수어 인코딩
            encoded_signs = self.encode_sign_sequence(sign_features, sign_lengths)
            
            # 텍스트 생성을 위한 초기 토큰
            generated_texts = []
            
            for batch_idx in range(batch_size):
                sign_repr = encoded_signs[batch_idx:batch_idx+1]  # (1, T_sign, dim)
                
                # BOS 토큰으로 시작
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
                
                for step in range(max_length):
                    # 현재까지 생성된 텍스트의 임베딩
                    if self.model_type == "gpt2":
                        text_embeddings = self.text_decoder.transformer.wte(input_ids)
                    else:
                        text_embeddings = self.text_decoder.shared(input_ids)
                    
                    text_projected = self.decoder_projection(text_embeddings)
                    
                    # Cross-attention
                    attended_features, _ = self.cross_attention(
                        query=text_projected,
                        key=sign_repr,
                        value=sign_repr
                    )
                    
                    # 특징 융합
                    fused_features = torch.cat([text_projected, attended_features], dim=-1)
                    enhanced_features = self.fusion_layer(fused_features)
                    
                    # 다음 토큰 예측
                    if self.model_type == "gpt2":
                        outputs = self.text_decoder(
                            inputs_embeds=text_embeddings + enhanced_features
                        )
                    else:
                        outputs = self.text_decoder(
                            inputs_embeds=text_embeddings + enhanced_features
                        )
                    
                    # 다음 토큰 샘플링
                    logits = outputs.logits[0, -1, :] / temperature
                    
                    # Top-p 샘플링
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -float('inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # EOS 토큰이면 종료
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # 디코딩
                generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return generated_texts


class SignToTextDataset(Dataset):
    """수어-텍스트 쌍 데이터셋"""
    
    def __init__(self, 
                 sign_data_path: str,
                 text_annotations_path: str,  # 수어에 대응하는 자연어 문장 파일
                 tokenizer,
                 max_sign_length: int = 32,
                 max_text_length: int = 50):
        
        # 수어 데이터 로드
        self.sign_dataset = SignLanguageDataset(sign_data_path, max_sign_length)
        
        # 텍스트 어노테이션 로드
        with open(text_annotations_path, 'r', encoding='utf-8') as f:
            self.text_annotations = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # 데이터 매칭 (비디오 경로를 키로 사용)
        self.matched_data = self._match_sign_text_data()
    
    def _match_sign_text_data(self) -> List[Dict]:
        """수어 데이터와 텍스트 어노테이션 매칭"""
        matched = []
        
        for idx, sign_sample in enumerate(self.sign_dataset):
            video_path = sign_sample['video_path']
            video_name = Path(video_path).stem
            
            # 해당 비디오의 텍스트 어노테이션 찾기
            if video_name in self.text_annotations:
                text = self.text_annotations[video_name]['sentence']
                matched.append({
                    'sign_idx': idx,
                    'text': text
                })
        
        return matched
    
    def __len__(self):
        return len(self.matched_data)
    
    def __getitem__(self, idx):
        data = self.matched_data[idx]
        
        # 수어 데이터
        sign_sample = self.sign_dataset[data['sign_idx']]
        
        # 텍스트 토크나이징
        text = data['text']
        encoded_text = self.tokenizer.encode(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'sign_features': sign_sample['features'],
            'sign_labels': sign_sample['labels'], 
            'text_tokens': encoded_text.squeeze(0),
            'text_string': text,
            'video_path': sign_sample['video_path']
        }


class SignToTextTrainer:
    """수어-텍스트 모델 훈련 클래스"""
    
    def __init__(self,
                 model: SignToTextModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 lr: float = 1e-4):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 옵티마이저 (인코더와 디코더에 다른 학습률 적용)
        encoder_params = list(self.model.sign_encoder.parameters())
        decoder_params = (list(self.model.text_decoder.parameters()) + 
                         list(self.model.cross_attention.parameters()) +
                         list(self.model.encoder_projection.parameters()) +
                         list(self.model.decoder_projection.parameters()) +
                         list(self.model.fusion_layer.parameters()))
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},  # 인코더는 낮은 학습률
            {'params': decoder_params, 'lr': lr}
        ])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            sign_features = batch['sign_features'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(
                sign_features=sign_features,
                target_texts=text_tokens
            )
            
            loss = outputs['loss']
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                sign_features = batch['sign_features'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                
                outputs = self.model(
                    sign_features=sign_features,
                    target_texts=text_tokens
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def train(self, num_epochs: int, save_dir: str) -> List[Dict]:
        """전체 훈련 프로세스"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 훈련
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step(val_metrics['val_loss'])
            
            epoch_metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            }
            history.append(epoch_metrics)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # 최고 모델 저장
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'tokenizer': self.model.tokenizer,
                    'epoch': epoch + 1,
                    'metrics': epoch_metrics
                }, save_path / 'best_model.pt', _use_new_zipfile_serialization=False)
        
        return history


def create_text_annotations_example():
    """텍스트 어노테이션 파일 예시 생성"""
    # 실제로는 AIHub 데이터의 JSON에서 자연어 문장을 추출해야 함
    example_annotations = {
        "video_001": {
            "sentence": "안녕하세요. 만나서 반갑습니다.",
            "keywords": ["안녕", "만나다", "반갑다"]
        },
        "video_002": {
            "sentence": "오늘 날씨가 정말 좋네요.",
            "keywords": ["오늘", "날씨", "좋다"]
        },
        # ... 더 많은 예시
    }
    
    with open('text_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(example_annotations, f, ensure_ascii=False, indent=2)
    
    return 'text_annotations.json'


def train_sign_to_text_model(
    sign_data_dir: str = "./processed_data",
    text_annotations_path: str = "./text_annotations.json", 
    save_dir: str = "./sign_to_text_models",
    pretrained_encoder_path: str = "./models/best_model.pt",
    text_model_name: str = "skt/kogpt2-base-v2",
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 30
):
    """수어-텍스트 모델 훈련 메인 함수"""
    
    device = setup_device()
    
    # 사전훈련된 수어 인코더 로드
    encoder_checkpoint = torch.load(pretrained_encoder_path, map_location=device, weights_only=False)
    encoder_config = encoder_checkpoint['model_config']
    
    sign_encoder = OpenHandsKoreanSignModel(
        input_dim=encoder_config['input_dim'],
        vocab_size=encoder_config['vocab_size'],
        d_model=encoder_config['d_model'],
        max_seq_length=encoder_config['max_seq_length']
    )
    sign_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    
    # 통합 모델 생성
    model = SignToTextModel(
        sign_encoder=sign_encoder,
        text_model_name=text_model_name,
        freeze_encoder=True  # 사전훈련된 인코더 동결
    )
    
    # 데이터셋 로드
    train_dataset = SignToTextDataset(
        sign_data_path=f"{sign_data_dir}/train_data.pt",
        text_annotations_path=text_annotations_path,
        tokenizer=model.tokenizer
    )
    
    val_dataset = SignToTextDataset(
        sign_data_path=f"{sign_data_dir}/val_data.pt", 
        text_annotations_path=text_annotations_path,
        tokenizer=model.tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 트레이너 생성 및 훈련
    trainer = SignToTextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=learning_rate
    )
    
    history = trainer.train(num_epochs=num_epochs, save_dir=save_dir)
    
    return history, model


if __name__ == "__main__":
    # 사용 예시
    
    # 1. 텍스트 어노테이션 파일 생성 (예시)
    text_annotations_path = create_text_annotations_example()
    
    # 2. 모델 훈련
    history, model = train_sign_to_text_model(
        sign_data_dir="./processed_data",
        text_annotations_path=text_annotations_path,
        pretrained_encoder_path="./models/best_model.pt",
        save_dir="./sign_to_text_models"
    )
    
    # 3. 추론 예시
    device = setup_device()
    model.eval()
    
    # 샘플 수어 특징 (실제로는 데이터에서 가져옴)
    sample_features = torch.randn(1, 32, 144).to(device)  # (1, seq_len, feature_dim)
    
    # 텍스트 생성
    generated_texts = model.generate_text(sample_features, max_length=30)
    print(f"생성된 텍스트: {generated_texts[0]}")
