import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import BeamSearchScorer, LogitsProcessorList, TopPLogitsWarper, TemperatureLogitsWarper


class AdvancedSignToTextModel(nn.Module):
    """
    고급 수어-텍스트 변환 모델 (향상된 기능들)
    """
    
    def __init__(self, base_model, **kwargs):
        super().__init__()
        self.base_model = base_model
        
        # 추가 구성요소들
        self.temporal_attention = TemporalAttentionModule(base_model.encoder_dim)
        self.context_memory = ContextMemoryModule(base_model.encoder_dim, memory_size=100)
        self.grammar_checker = GrammarAugmentedDecoder(base_model.tokenizer)
    
    def forward(self, *args, **kwargs):
        # 기본 모델에 고급 기능들을 추가
        outputs = self.base_model(*args, **kwargs)
        
        if 'encoded_signs' in outputs:
            # 시간적 패턴 강화
            enhanced_signs = self.temporal_attention(outputs['encoded_signs'])
            
            # 문맥 메모리 업데이트
            self.context_memory.update(enhanced_signs)
            
            outputs['enhanced_signs'] = enhanced_signs
        
        return outputs


class TemporalAttentionModule(nn.Module):
    """시간적 패턴을 고려한 어텐션 모듈"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        
        # 1D 컨볼루션으로 시간적 패턴 추출
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Self-attention으로 시간적 의존성 모델링
        attended_x, _ = self.temporal_attention(x_conv, x_conv, x_conv)
        
        return x + attended_x  # Residual connection


class ContextMemoryModule(nn.Module):
    """이전 문맥을 기억하는 메모리 모듈"""
    
    def __init__(self, hidden_dim: int, memory_size: int = 100):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        
        # 메모리 뱅크
        self.register_buffer('memory_bank', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # 메모리 어텐션
        self.memory_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    
    def update(self, features: torch.Tensor):
        """메모리 업데이트"""
        # 배치의 평균 특징을 메모리에 저장
        batch_feature = features.mean(dim=(0, 1))  # (H,)
        
        ptr = self.memory_ptr.item()
        self.memory_bank[ptr] = batch_feature
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def retrieve_context(self, query: torch.Tensor) -> torch.Tensor:
        """관련 문맥 검색"""
        # query: (B, T, H)
        memory = self.memory_bank.unsqueeze(0).expand(query.size(0), -1, -1)  # (B, M, H)
        
        context, _ = self.memory_attention(query, memory, memory)
        return context


class GrammarAugmentedDecoder(nn.Module):
    """문법 규칙을 고려한 디코더"""
    
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        
        # 한국어 문법 패턴 정의
        self.grammar_patterns = {
            'subject_markers': ['이', '가', '은', '는'],
            'object_markers': ['을', '를'],
            'verb_endings': ['다', '요', '습니다', '네요'],
            'connecting_endings': ['고', '서', '며', '면서']
        }
        
        # 품사 태거 (실제로는 KoNLPy 등 사용)
        self.pos_classifier = nn.Linear(768, len(self.grammar_patterns))
    
    def apply_grammar_constraints(self, logits: torch.Tensor, 
                                generated_tokens: List[int]) -> torch.Tensor:
        """문법 제약 조건 적용"""
        if len(generated_tokens) < 2:
            return logits
        
        # 간단한 규칙 기반 제약
        last_token = self.tokenizer.decode([generated_tokens[-1]])
        
        # 예: 조사 다음에 다시 조사가 오지 않도록
        if last_token in self.grammar_patterns['subject_markers']:
            for marker in self.grammar_patterns['subject_markers']:
                marker_id = self.tokenizer.encode(marker, add_special_tokens=False)
                if marker_id:
                    logits[marker_id[0]] -= 10.0  # 확률 감소
        
        return logits


class BeamSearchDecoder:
    """빔 서치 기반 디코딩"""
    
    def __init__(self, model, tokenizer, beam_size: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
    
    def generate_with_beam_search(self, 
                                 sign_features: torch.Tensor,
                                 max_length: int = 50,
                                 length_penalty: float = 1.0) -> List[str]:
        """빔 서치로 텍스트 생성"""
        device = sign_features.device
        batch_size = sign_features.size(0)
        
        # 수어 인코딩
        with torch.no_grad():
            encoded_signs = self.model.encode_sign_sequence(sign_features)
        
        results = []
        
        for batch_idx in range(batch_size):
            sign_repr = encoded_signs[batch_idx:batch_idx+1]
            
            # 빔 서치 초기화
            beams = [(torch.tensor([[self.tokenizer.bos_token_id]], device=device), 0.0)]
            
            for step in range(max_length):
                candidates = []
                
                for seq, score in beams:
                    if seq[0, -1].item() == self.tokenizer.eos_token_id:
                        candidates.append((seq, score))
                        continue
                    
                    # 다음 토큰 확률 계산
                    with torch.no_grad():
                        if self.model.model_type == "gpt2":
                            text_embeddings = self.model.text_decoder.transformer.wte(seq)
                        else:
                            text_embeddings = self.model.text_decoder.shared(seq)
                        
                        text_projected = self.model.decoder_projection(text_embeddings)
                        attended_features, _ = self.model.cross_attention(
                            query=text_projected, key=sign_repr, value=sign_repr
                        )
                        
                        fused_features = torch.cat([text_projected, attended_features], dim=-1)
                        enhanced_features = self.model.fusion_layer(fused_features)
                        
                        if self.model.model_type == "gpt2":
                            outputs = self.model.text_decoder(
                                inputs_embeds=text_embeddings + enhanced_features
                            )
                        else:
                            outputs = self.model.text_decoder(
                                inputs_embeds=text_embeddings + enhanced_features
                            )
                        
                        logits = outputs.logits[0, -1, :]
                        log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Top-k 후보 생성
                    top_k_probs, top_k_ids = torch.topk(log_probs, self.beam_size)
                    
                    for prob, token_id in zip(top_k_probs, top_k_ids):
                        new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + prob.item()
                        
                        # 길이 보정
                        length_corrected_score = new_score / (new_seq.size(1) ** length_penalty)
                        candidates.append((new_seq, length_corrected_score))
                
                # 상위 빔 선택
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_size]
                
                # 모든 빔이 EOS에 도달했으면 종료
                if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in beams):
                    break
            
            # 최고 스코어 시퀀스 선택
            best_seq = beams[0][0]
            generated_text = self.tokenizer.decode(best_seq[0], skip_special_tokens=True)
            results.append(generated_text)
        
        return results


class MultiModalAttention(nn.Module):
    """다중 모달리티 어텐션 (수어 + 얼굴 표정 + 몸짓)"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hand_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.face_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.pose_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 모달리티 융합
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, hand_features: torch.Tensor,
                face_features: torch.Tensor,
                pose_features: torch.Tensor) -> torch.Tensor:
        
        # 각 모달리티별 self-attention
        hand_attended, _ = self.hand_attention(hand_features, hand_features, hand_features)
        face_attended, _ = self.face_attention(face_features, face_features, face_features)  
        pose_attended, _ = self.pose_attention(pose_features, pose_features, pose_features)
        
        # 가중합 융합
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = (weights[0] * hand_attended + 
                         weights[1] * face_attended + 
                         weights[2] * pose_attended)
        
        return fused_features


class EmotionAwareDecoder(nn.Module):
    """감정을 고려한 디코더"""
    
    def __init__(self, hidden_dim: int, num_emotions: int = 7):
        super().__init__()
        
        # 감정 분류기 (기쁨, 슬픔, 화남, 놀람, 두려움, 혐오, 중립)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
        
        # 감정별 디코딩 스타일
        self.emotion_embeddings = nn.Embedding(num_emotions, hidden_dim)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 감정 예측
        emotion_logits = self.emotion_classifier(features.mean(dim=1))  # (B, num_emotions)
        predicted_emotion = emotion_logits.argmax(dim=-1)  # (B,)
        
        # 감정 임베딩 추가
        emotion_emb = self.emotion_embeddings(predicted_emotion).unsqueeze(1)  # (B, 1, H)
        enhanced_features = features + emotion_emb
        
        return enhanced_features, emotion_logits


class InteractiveSignTranslator:
    """실시간 상호작용 수어 번역기"""
    
    def __init__(self, model_path: str):
        self.device = setup_device()
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = checkpoint['model']
        self.model.eval()
        
        # 빔 서치 디코더
        self.beam_decoder = BeamSearchDecoder(self.model, self.model.tokenizer)
        
        # 대화 상태 관리
        self.conversation_history = []
        self.context_window = 5  # 최근 5개 문장만 고려
    
    def translate_sign_sequence(self, 
                              sign_features: np.ndarray,
                              use_beam_search: bool = True,
                              use_context: bool = True) -> Dict:
        """수어 시퀀스를 텍스트로 번역"""
        
        features_tensor = torch.FloatTensor(sign_features).unsqueeze(0).to(self.device)
        
        # 1. 기본 번역
        if use_beam_search:
            translations = self.beam_decoder.generate_with_beam_search(
                features_tensor, max_length=50, length_penalty=1.2
            )
            primary_translation = translations[0]
        else:
            translations = self.model.generate_text(features_tensor, max_length=50)
            primary_translation = translations[0]
        
        # 2. 문맥 기반 후처리
        if use_context and len(self.conversation_history) > 0:
            primary_translation = self._apply_context_correction(primary_translation)
        
        # 3. 대화 히스토리 업데이트
        self.conversation_history.append({
            'timestamp': time.time(),
            'translation': primary_translation,
            'confidence': self._calculate_confidence(features_tensor)
        })
        
        # 4. 컨텍스트 윈도우 유지
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
        
        return {
            'translation': primary_translation,
            'confidence': self.conversation_history[-1]['confidence'],
            'context': self._get_conversation_context(),
            'suggestions': self._generate_alternative_translations(features_tensor)
        }
    
    def _apply_context_correction(self, translation: str) -> str:
        """문맥을 고려한 번역 수정"""
        # 이전 대화와의 연결성 확인
        recent_translations = [h['translation'] for h in self.conversation_history[-3:]]
        
        # 간단한 규칙 기반 수정
        if any('안녕' in t for t in recent_translations) and '안녕' in translation:
            translation = translation.replace('안녕하세요', '네, 안녕하세요')
        
        return translation
    
    def _calculate_confidence(self, features: torch.Tensor) -> float:
        """번역 신뢰도 계산"""
        with torch.no_grad():
            outputs = self.model(features)
            if 'attention_weights' in outputs:
                # 어텐션 가중치의 엔트로피로 신뢰도 추정
                attention_entropy = torch.distributions.Categorical(
                    probs=outputs['attention_weights']
                ).entropy().mean().item()
                confidence = 1.0 / (1.0 + attention_entropy)
            else:
                confidence = 0.8  # 기본값
        
        return float(confidence)
    
    def _get_conversation_context(self) -> str:
        """대화 문맥 요약"""
        if len(self.conversation_history) <= 1:
            return "새로운 대화 시작"
        
        recent_topics = []
        for entry in self.conversation_history[-3:]:
            words = entry['translation'].split()
            if words:
                recent_topics.extend(words[:2])  # 각 문장에서 처음 2단어만
        
        return f"최근 주제: {', '.join(set(recent_topics))}"
    
    def _generate_alternative_translations(self, features: torch.Tensor) -> List[str]:
        """대안 번역 생성"""
        alternatives = []
        
        # 다른 디코딩 파라미터로 여러 번역 생성
        for temp in [0.6, 1.0, 1.4]:
            alt_translations = self.model.generate_text(
                features, max_length=40, temperature=temp
            )
            if alt_translations[0] not in alternatives:
                alternatives.append(alt_translations[0])
        
        return alternatives[:3]  # 최대 3개 대안


# 사용 예시 함수
def setup_advanced_training():
    """고급 모델 훈련 설정"""
    
    # 기본 모델 로드
    base_model = SignToTextModel(...)  # 이전에 정의된 모델
    
    # 고급 기능 추가
    advanced_model = AdvancedSignToTextModel(base_model)
    
    return advanced_model


def create_interactive_demo():
    """대화형 데모 생성"""
    
    translator = InteractiveSignTranslator("./sign_to_text_models/best_model.pt")
    
    print("수어 번역기가 준비되었습니다!")
    print("수어 영상을 입력하면 실시간으로 번역합니다.\n")
    
    while True:
        # 실제로는 웹캠이나 비디오 파일에서 수어 특징 추출
        sample_features = np.random.randn(32, 144)  # 예시 데이터
        
        result = translator.translate_sign_sequence(
            sample_features, 
            use_beam_search=True,
            use_context=True
        )
        
        print(f"번역 결과: {result['translation']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"문맥: {result['context']}")
        print(f"대안: {', '.join(result['suggestions'])}")
        print("-" * 50)
        
        user_input = input("계속하려면 Enter, 종료하려면 'q': ")
        if user_input.lower() == 'q':
            break


if __name__ == "__main__":
    # 대화형 데모 실행
    create_interactive_demo()