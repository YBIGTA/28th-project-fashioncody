# 28th-project-fashioncody
28기 신입기수 프로젝트 - 멀티모달 코디네이터


## 비전팀


### 이미지 배경 제거 실행
1. Environment Setup

       pip install -r requirements.txt

2. Execution

        python process_clothing_images.py --input_dir data --output_dir processed

### 파이프라인 실행 (YOLO -> 이미지 제거 -> CLIP)
1. Environment Setup

       pip install -r requirements.txt

2. Execution
   
       python pipeline.py --image "옷사진.jpg" --model best.pt
       python pipeline.py --image_dir "이미지폴더/" --model best.pt --output results.json
