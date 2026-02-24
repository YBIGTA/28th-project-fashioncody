"""
토크나이저 설정
"""
from transformers import AutoTokenizer

def get_tokenizer(model_name: str = 'monologg/kobert'):
    """
    한국어 토크나이저 로드
    
    옵션:
    - 'monologg/kobert': KoBERT (BERT 기반)
    - 'monologg/koelectra-base-v3-discriminator': KoELECTRA (더 빠름)
    - 'skt/kogpt2-base-v2': KoGPT-2 (생성 모델)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# 사용 예시
if __name__ == '__main__':
    tokenizer = get_tokenizer()
    text = "미니멀 데이트 코디"
    encoded = tokenizer(text, return_tensors='pt')
    print(f"Input: {text}")
    print(f"Token IDs: {encoded['input_ids']}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
