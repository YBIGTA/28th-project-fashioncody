"""
EfficientNet 기반 의류 속성 분류기 (12개 속성)

학습된 efficientnet_kfashion_best.pt 모델로 의류 이미지에서
12개 속성을 예측하고 날씨 범주를 자동 추론합니다.

속성 (12개):
  단일라벨: 카테고리, 색상, 서브색상, 소매기장, 기장, 핏, 옷깃, 스타일, 서브스타일
  다중라벨: 소재, 프린트, 디테일
  자동추론: 날씨 (7단계, -20~40°C)
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


# ============================================================
# 속성 정의 (학습 시와 동일해야 함)
# ============================================================

SINGLE_LABEL_ATTRS = {
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

MULTI_LABEL_ATTRS = {
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

# 하의는 소매기장/옷깃 미사용
NO_SLEEVE_COLLAR_PARTS = {'bottom', '하의'}


# ============================================================
# 날씨 범주 자동 추론 (colab_create_structured_data.ipynb 동일)
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
    """
    의류 속성 기반 날씨 범주 자동 추론 (-20~40°C, 7단계).

    Args:
        category: 카테고리 예측값 (예: '후드티', '패딩')
        sleeve_length: 소매기장 예측값 (예: '긴팔', '반팔')
        materials: 소재 예측값 리스트 (예: ['니트', '울/캐시미어'])
        length: 기장 예측값 (예: '롱', '크롭')

    Returns:
        str: 날씨 범주 (한파/한겨울/쌀쌀/선선/따뜻/더움/폭염)
    """
    score = 0
    score += CATEGORY_SCORE.get(category, 0)
    score += SLEEVE_SCORE.get(sleeve_length, 0)

    if materials:
        mat_scores = [MATERIAL_SCORE.get(m, 0) for m in materials if m]
        if mat_scores:
            score += max(mat_scores)

    score += LENGTH_SCORE.get(length, 0)

    if score >= 7:
        return '한파'      # -20~-5°C
    elif score >= 5:
        return '한겨울'    # -5~5°C
    elif score >= 3:
        return '쌀쌀'      # 5~15°C
    elif score >= 1:
        return '선선'      # 15~20°C
    elif score >= -1:
        return '따뜻'      # 20~25°C
    elif score >= -3:
        return '더움'      # 25~33°C
    else:
        return '폭염'      # 33~40°C


# ============================================================
# EfficientNet Multi-Task 모델 구조 (학습 시와 동일)
# ============================================================

class FashionMultiTaskModel(nn.Module):
    def __init__(self, single_label_attrs, multi_label_attrs,
                 backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features  # 1280

        self.dropout = nn.Dropout(0.3)

        self.single_heads = nn.ModuleDict()
        for attr_name, labels in single_label_attrs.items():
            self.single_heads[attr_name] = nn.Linear(feat_dim, len(labels))

        self.multi_heads = nn.ModuleDict()
        for attr_name, labels in multi_label_attrs.items():
            self.multi_heads[attr_name] = nn.Linear(feat_dim, len(labels))

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        outputs = {}
        for attr_name, head in self.single_heads.items():
            outputs[attr_name] = head(features)
        for attr_name, head in self.multi_heads.items():
            outputs[attr_name] = head(features)
        return outputs


# ============================================================
# EfficientNet 분류기
# ============================================================

# 추론용 이미지 전처리 (학습 시 val_transform과 동일)
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# YOLO 영어 카테고리 → 한국어 매핑
YOLO_TO_KR = {
    'top':    '상의',
    'bottom': '하의',
    'outer':  '아우터',
    'dress':  '원피스',
    'acc':    '액세서리',
}


class EfficientNetClassifier:
    """
    학습된 EfficientNet Multi-Task 모델로 의류 속성 분류.

    사용법:
        classifier = EfficientNetClassifier('src/recognition/models/efficientnet_kfashion_best.pt')
        result = classifier.classify(pil_image, yolo_category='top')
    """

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: efficientnet_kfashion_best.pt 경로
            device: 'cuda' / 'cpu' (None이면 자동)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 체크포인트 로드
        ckpt = torch.load(model_path, map_location=self.device)

        # 속성 정의를 체크포인트에서 읽거나 기본값 사용
        self.single_attrs = ckpt.get('single_label_attrs', SINGLE_LABEL_ATTRS)
        self.multi_attrs  = ckpt.get('multi_label_attrs', MULTI_LABEL_ATTRS)

        # 모델 생성 및 가중치 로드
        self.model = FashionMultiTaskModel(self.single_attrs, self.multi_attrs)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f'EfficientNetClassifier 로드 완료 (device={self.device})')

    @torch.no_grad()
    def classify(self, image, yolo_category='top'):
        """
        PIL 이미지 분류 → dummy.json 형식으로 반환.

        Args:
            image: PIL.Image (크롭된 의류 이미지)
            yolo_category: YOLO 탐지 카테고리 ('top'/'bottom'/'outer'/'dress'/'acc')

        Returns:
            dict: dummy.json 형식의 속성 딕셔너리
                  (image_name, category, detection_confidence는 pipeline에서 추가)
        """
        # 전처리
        img_tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(self.device)

        # 추론
        outputs = self.model(img_tensor)

        result = {}
        is_bottom = yolo_category in NO_SLEEVE_COLLAR_PARTS

        # --- 단일 라벨 속성 ---
        for attr_name, labels in self.single_attrs.items():
            logits = outputs[attr_name].squeeze()
            probs  = torch.softmax(logits, dim=0)
            idx    = probs.argmax().item()
            value  = labels[idx]
            conf   = round(probs[idx].item(), 4)

            # 하의는 소매기장/옷깃 제외
            if is_bottom and attr_name in ('소매기장', '옷깃'):
                continue

            field, conf_field = self._field_name(attr_name)
            result[field]      = value
            result[conf_field] = conf

        # --- 다중 라벨 속성 ---
        material_values = []
        for attr_name, labels in self.multi_attrs.items():
            logits = outputs[attr_name].squeeze()
            probs  = torch.sigmoid(logits)

            positives = []
            for label, prob in zip(labels, probs):
                if prob.item() > 0.5:
                    positives.append({'value': label,
                                      'confidence': round(prob.item(), 4)})
            if not positives:
                positives = [{'value': '없음', 'confidence': 0.0}]

            field = self._multi_field_name(attr_name)
            result[field] = positives

            if attr_name == '소재':
                material_values = [p['value'] for p in positives if p['value'] != '없음']

        # --- 날씨 자동 추론 ---
        sub_type      = result.get('sub_type', '')
        sleeve_length = result.get('sleeve_length', '없음')
        length        = result.get('length', '노멀')
        result['weather'] = assign_weather(sub_type, sleeve_length,
                                           material_values, length)

        return result

    # ---- 내부 헬퍼 ----

    def _field_name(self, attr_name):
        """한국어 속성명 → (영어 필드명, 영어 confidence 필드명)"""
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
        """한국어 다중라벨 속성명 → 영어 필드명"""
        mapping = {
            '소재':  'material',
            '프린트': 'print',
            '디테일': 'detail',
        }
        return mapping.get(attr_name, attr_name)
