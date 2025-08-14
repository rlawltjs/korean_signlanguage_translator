import torch

# 모델 파일 로드
model_path = "model.pt"  # 모델 파일 경로를 지정하세요
model_data = torch.load(model_path, map_location=torch.device('cpu'))

# 모델 구조와 매개변수 확인
print("모델 키:", model_data.keys())

# 만약 모델에 클래스 매핑 정보가 저장되어 있다면:
if 'class_to_idx' in model_data:
    class_mapping = model_data['class_to_idx']
    print("\n클래스 매핑:")
    for class_name, idx in class_mapping.items():
        print(f"{idx}: {class_name}")

# 모델의 구조 확인
if 'state_dict' in model_data:
    for key in model_data['state_dict'].keys():
        print(f"\n레이어: {key}")
