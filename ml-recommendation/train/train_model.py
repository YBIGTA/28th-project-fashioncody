"""
코디 추천 MLP 모델 학습 스크립트
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import json
import os

class OutfitDataset(Dataset):
    """코디 데이터셋"""
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 텍스트 토크나이징
        mood_text = str(item.get('mood', ''))
        comment_text = str(item.get('comment', ''))
        combined_text = f"{mood_text} {comment_text}"
        
        encoded = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 수치 특징
        features = torch.tensor([
            item.get('temperature', 0),
            item.get('feels_like', 0),
            item.get('precipitation', 0),
            item.get('top_category_idx', 0),
            item.get('bottom_category_idx', 0),
            item.get('outer_category_idx', 0),
            # 추가 특징들...
        ], dtype=torch.float32)
        
        # 라벨 (매칭 점수)
        label = torch.tensor(item.get('score', 0.0), dtype=torch.float32)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'features': features,
            'label': label
        }

class OutfitRecommendationModel(nn.Module):
    """코디 추천 MLP 모델"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dims=[256, 128, 64], num_features=7):
        super().__init__()
        
        # 텍스트 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 텍스트 인코더 (간단한 LSTM 또는 Transformer)
        self.text_encoder = nn.LSTM(
            embedding_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True
        )
        
        # 특징 결합
        text_dim = 128 * 2  # bidirectional LSTM
        combined_dim = text_dim + num_features
        
        # MLP 레이어
        layers = []
        input_dim = combined_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        # 출력 레이어 (매칭 점수 0-1)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask, features):
        # 텍스트 임베딩
        embedded = self.embedding(input_ids)
        
        # LSTM 인코딩
        lstm_out, (hidden, _) = self.text_encoder(embedded)
        # 마지막 hidden state 사용
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # 특징 결합
        combined = torch.cat([text_features, features], dim=1)
        
        # MLP 통과
        output = self.mlp(combined)
        
        return output.squeeze()

def train_model():
    """모델 학습"""
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # 토크나이저 로드 (한국어 지원)
    tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
    # 또는 'monologg/koelectra-base-v3-discriminator'
    
    # 데이터 로드
    # TODO: 실제 데이터셋 경로로 변경
    df = pd.read_csv('data/train.csv')
    
    # 데이터 전처리
    # TODO: 카테고리 인덱싱, 정규화 등
    
    # Train/Val 분할
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 데이터셋 생성
    train_dataset = OutfitDataset(train_df, tokenizer)
    val_dataset = OutfitDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델 초기화
    vocab_size = len(tokenizer.vocab)
    model = OutfitRecommendationModel(vocab_size=vocab_size).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/best_model.pt')
            tokenizer.save_pretrained('../models/')
    
    print("학습 완료!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    train_model()
