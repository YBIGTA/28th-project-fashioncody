"""
데이터 전처리 스크립트
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

def preprocess_data(raw_data_path, output_path):
    """
    원본 데이터를 모델 학습용으로 전처리
    
    Args:
        raw_data_path: 원본 CSV 파일 경로
        output_path: 전처리된 데이터 저장 경로
    """
    df = pd.read_csv(raw_data_path)
    
    # 카테고리 인코딩
    category_encoder = LabelEncoder()
    categories = ['top', 'bottom', 'outer', 'shoes', 'bag', 'accessory']
    category_encoder.fit(categories)
    
    # 색상 인코딩 (간단한 예시)
    color_encoder = LabelEncoder()
    all_colors = df['top_color'].dropna().tolist() + \
                 df['bottom_color'].dropna().tolist() + \
                 df['outer_color'].dropna().tolist()
    color_encoder.fit(list(set(all_colors)))
    
    # 특징 추출
    processed_data = []
    
    for _, row in df.iterrows():
        # 카테고리 인덱스
        top_cat_idx = category_encoder.transform([row['top_category']])[0] if pd.notna(row['top_category']) else 0
        bottom_cat_idx = category_encoder.transform([row['bottom_category']])[0] if pd.notna(row['bottom_category']) else 0
        outer_cat_idx = category_encoder.transform([row['outer_category']])[0] if pd.notna(row['outer_category']) else 0
        
        # 시즌 one-hot
        seasons = ['spring', 'summer', 'fall', 'winter']
        season_vector = [1 if s in str(row.get('season', '')) else 0 for s in seasons]
        
        processed_row = {
            'mood': row.get('mood', ''),
            'comment': row.get('comment', ''),
            'temperature': float(row.get('temperature', 0)),
            'feels_like': float(row.get('feels_like', 0)),
            'precipitation': float(row.get('precipitation', 0)),
            'top_category_idx': top_cat_idx,
            'bottom_category_idx': bottom_cat_idx,
            'outer_category_idx': outer_cat_idx,
            'score': float(row.get('score', 0.0)),  # 라벨
        }
        
        processed_data.append(processed_row)
    
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_path, index=False)
    
    # 인코더 저장
    encoders = {
        'category_encoder': {cat: int(idx) for idx, cat in enumerate(categories)},
        'color_encoder': {color: int(idx) for idx, color in enumerate(color_encoder.classes_)}
    }
    
    with open('encoders.json', 'w', encoding='utf-8') as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)
    
    print(f"전처리 완료: {len(processed_df)}개 샘플")
    return processed_df

if __name__ == '__main__':
    # 예시 사용
    preprocess_data('data/raw_data.csv', 'data/train.csv')
