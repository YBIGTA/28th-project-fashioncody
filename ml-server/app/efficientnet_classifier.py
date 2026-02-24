"""
EfficientNet 기반 의류 속성 분류기 (12개 속성) — ONNX Runtime 추론

학습된 efficientnet_kfashion.onnx 모델로 의류 이미지에서
12개 속성을 예측하고 날씨 범주를 자동 추론합니다.

속성 (12개):
  단일라벨: 카테고리, 색상, 서브색상, 소매기장, 기장, 핏, 옷깃, 스타일, 서브스타일
  다중라벨: 소재, 프린트, 디테일
  자동추론: 날씨 (7단계, -20~40°C)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# 속성 정의 (기본값; effnet_labels.json에서 오버라이드 가능)
# ============================================================

SINGLE_LABEL_ATTRS: Dict[str, List[str]] = {
    '카테고리': ['가디건', '니트웨어', '드레스', '래깅스', '베스트', '브라탑',
               '블라우스', '셔츠', '스커트', '재킷', '점퍼', '점프수트',
               '조거팬츠', '짚업', '청바지', '코트', '탑', '티셔츠',
               '패딩', '팬츠', '후드티'],
    '색상': ['골드', '그레이', '그린', '네온', '네이비', '라벤더', '레드',
            '민트', '베이지', '브라운', '블랙', '블루', '스카이블루', '실버',
            '옐로우', '오렌지', '와인', '카키', '퍼플', '핑크', '화이트'],
    '서브색상': ['골드', '그레이', '그린', '네온', '네이비', '라벤더', '레드',
              '민트', '베이지', '브라운', '블랙', '블루', '스카이블루', '실버',
              '옐로우', '오렌지', '와인', '카키', '퍼플', '핑크', '화이트'],
    '소매기장': ['7부소매', '긴팔', '민소매', '반팔', '없음', '캡'],
    '기장': ['노멀', '니렝스', '롱', '맥시', '미니', '미디', '발목', '크롭', '하프'],
    '핏': ['노멀', '루즈', '벨보텀', '스키니', '오버사이즈', '와이드', '타이트'],
    '옷깃': ['밴드칼라', '보우칼라', '세일러칼라', '셔츠칼라', '숄칼라',
            '차이나칼라', '테일러드칼라', '폴로칼라', '피터팬칼라'],
    '스타일': ['레트로', '로맨틱', '리조트', '매니시', '모던', '밀리터리',
             '섹시', '소피스트케이티드', '스트리트', '스포티', '아방가르드',
             '오리엔탈', '웨스턴', '젠더리스', '컨트리', '클래식', '키치',
             '톰보이', '펑크', '페미닌', '프레피', '히피', '힙합'],
    '서브스타일': ['레트로', '로맨틱', '리조트', '매니시', '모던', '밀리터리',
               '섹시', '소피스트케이티드', '스트리트', '스포티', '아방가르드',
               '오리엔탈', '웨스턴', '젠더리스', '컨트리', '클래식', '키치',
               '톰보이', '펑크', '페미닌', '프레피', '히피', '힙합'],
}

MULTI_LABEL_ATTRS: Dict[str, List[str]] = {
    '소재': ['가죽', '네오프렌', '니트', '데님', '레이스', '린넨', '메시',
            '무스탕', '벨벳', '비닐/PVC', '스웨이드', '스판덱스', '시퀸/글리터',
            '시폰', '실크', '우븐', '울/캐시미어', '자카드', '저지', '코듀로이',
            '트위드', '패딩', '퍼', '플리스', '헤어 니트'],
    '프린트': ['그라데이션', '그래픽', '깅엄', '도트', '레터링', '무지', '믹스',
             '뱀피', '스트라이프', '아가일', '지그재그', '지브라', '체크',
             '카무플라쥬', '타이다이', '페이즐리', '플로럴', '하운즈투스',
             '하트', '해골', '호피'],
    '디테일': ['X스트랩', '글리터', '니트꽈베기', '단추', '더블브레스티드',
             '드롭숄더', '드롭웨이스트', '디스트로이드', '띠', '러플', '레이스',
             '레이스업', '롤업', '리본', '버클', '비대칭', '비즈', '셔링',
             '스터드', '스트링', '스티치', '스팽글', '슬릿', '싱글브레스티드',
             '자수', '지퍼', '체인', '컷아웃', '퀄팅', '태슬', '패치워크',
             '퍼트리밍', '퍼프', '페플럼', '포켓', '폼폼', '프린지', '프릴',
             '플레어', '플리츠'],
}

NO_SLEEVE_COLLAR_PARTS = {'bottom', '하의'}

# YOLO 영어 카테고리 → 한국어 매핑
YOLO_TO_KR = {
    'top':    '상의',
    'bottom': '하의',
    'outer':  '아우터',
    'dress':  '원피스',
    'acc':    '액세서리',
}


# ============================================================
# 날씨 범주 자동 추론
# ============================================================

CATEGORY_SCORE = {
    '패딩': 5, '코트': 4, '점퍼': 3, '짚업': 2, '후드티': 2,
    '니트웨어': 2, '가디건': 1, '재킷': 1, '베스트': 0,
    '셔츠': 0, '블라우스': -1, '티셔츠': -1,
    '탑': -2, '브라탑': -4,
    '드레스': 0, '점프수트': 0, '스커트': -1,
    '팬츠': 0, '청바지': 0, '조거팬츠': 0, '래깅스': 0,
}

SLEEVE_SCORE = {
    '긴팔': 1, '7부소매': 0, '반팔': -1, '캡': -2, '민소매': -3, '없음': 0,
}

MATERIAL_SCORE = {
    '퍼': 4, '무스탕': 4, '패딩': 4,
    '울/캐시미어': 3, '플리스': 3, '헤어 니트': 3,
    '트위드': 2, '코듀로이': 1, '니트': 1, '벨벳': 1,
    '우븐': 0, '데님': 0, '저지': 0, '자카드': 0, '스판덱스': 0,
    '가죽': 1, '스웨이드': 1, '네오프렌': 0,
    '레이스': -1, '실크': -1, '시퀸/글리터': 0,
    '시폰': -2, '린넨': -3, '메시': -3, '비닐/PVC': -1,
}

LENGTH_SCORE = {
    '맥시': 1, '롱': 1, '니렝스': 0, '미디': 0,
    '노멀': 0, '하프': 0, '발목': 0,
    '미니': -1, '크롭': -2,
}


def assign_weather(category, sleeve_length, materials, length):
    score = 0
    score += CATEGORY_SCORE.get(category, 0)
    score += SLEEVE_SCORE.get(sleeve_length, 0)
    if materials:
        mat_scores = [MATERIAL_SCORE.get(m, 0) for m in materials if m]
        if mat_scores:
            score += max(mat_scores)
    score += LENGTH_SCORE.get(length, 0)

    if score >= 7:
        return '한파'
    elif score >= 5:
        return '한겨울'
    elif score >= 3:
        return '쌀쌀'
    elif score >= 1:
        return '선선'
    elif score >= -1:
        return '따뜻'
    elif score >= -3:
        return '더움'
    else:
        return '폭염'


# ============================================================
# Numpy preprocessing (replaces torchvision.transforms)
# ============================================================

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """PIL Image -> NCHW float32 numpy array (ImageNet normalized)."""
    img = image.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0       # HWC [0,1]
    arr = (arr - _MEAN) / _STD                           # normalize
    arr = arr.transpose(2, 0, 1)                         # HWC -> CHW
    return np.expand_dims(arr, axis=0)                   # NCHW


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# EfficientNet 분류기 (ONNX Runtime)
# ============================================================

class EfficientNetClassifier:
    def __init__(self, model_path: str, labels_path: Optional[str] = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Load label definitions
        if labels_path is None:
            labels_path = str(Path(model_path).parent / "effnet_labels.json")

        if Path(labels_path).exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            self.single_attrs = labels.get("single_label_attrs", SINGLE_LABEL_ATTRS)
            self.multi_attrs = labels.get("multi_label_attrs", MULTI_LABEL_ATTRS)
        else:
            self.single_attrs = SINGLE_LABEL_ATTRS
            self.multi_attrs = MULTI_LABEL_ATTRS

        print(f'EfficientNetClassifier loaded (ONNX Runtime)')

    def classify(self, image: Image.Image, yolo_category: str = 'top') -> dict:
        input_data = _preprocess_image(image)
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_data})
        outputs = dict(zip(self.output_names, raw_outputs))

        result = {}
        is_bottom = yolo_category in NO_SLEEVE_COLLAR_PARTS

        # --- 단일 라벨 속성 ---
        for attr_name, labels in self.single_attrs.items():
            logits = outputs[attr_name].squeeze()
            probs = _softmax(logits)
            idx = int(np.argmax(probs))
            value = labels[idx]
            conf = round(float(probs[idx]), 4)

            if is_bottom and attr_name in ('소매기장', '옷깃'):
                continue

            field, conf_field = self._field_name(attr_name)
            result[field] = value
            result[conf_field] = conf

        # --- 다중 라벨 속성 ---
        material_values = []
        for attr_name, labels in self.multi_attrs.items():
            logits = outputs[attr_name].squeeze()
            probs = _sigmoid(logits)

            positives = []
            for label, prob in zip(labels, probs):
                if float(prob) > 0.5:
                    positives.append({'value': label, 'confidence': round(float(prob), 4)})
            if not positives:
                positives = [{'value': '없음', 'confidence': 0.0}]

            field = self._multi_field_name(attr_name)
            result[field] = positives

            if attr_name == '소재':
                material_values = [p['value'] for p in positives if p['value'] != '없음']

        # --- 날씨 자동 추론 ---
        sub_type = result.get('sub_type', '')
        sleeve_length = result.get('sleeve_length', '없음')
        length = result.get('length', '노멀')
        result['weather'] = assign_weather(sub_type, sleeve_length, material_values, length)

        return result

    def _field_name(self, attr_name):
        mapping = {
            '카테고리':  ('sub_type',      'sub_type_confidence'),
            '색상':      ('color',         'color_confidence'),
            '서브색상':  ('sub_color',     'sub_color_confidence'),
            '소매기장':  ('sleeve_length', 'sleeve_length_confidence'),
            '기장':      ('length',        'length_confidence'),
            '핏':        ('fit',           'fit_confidence'),
            '옷깃':      ('collar',        'collar_confidence'),
            '스타일':    ('style',         'style_confidence'),
            '서브스타일':('sub_style',     'sub_style_confidence'),
        }
        return mapping.get(attr_name, (attr_name, f'{attr_name}_confidence'))

    def _multi_field_name(self, attr_name):
        mapping = {
            '소재':  'material',
            '프린트': 'print',
            '디테일': 'detail',
        }
        return mapping.get(attr_name, attr_name)
