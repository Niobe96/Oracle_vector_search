import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import pickle
from tqdm import tqdm
import shutil
from glob import glob
import yaml

# --- ⚙️ 1. 설정 (Configuration) ---
# 스크립트가 실행되는 위치를 기준으로 경로 설정
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 입력 경로
ORIGINAL_BASE_DIR = os.path.join(BASE_PROJECT_DIR, 'data', 'raw', 'Brain_MRI')

# ✅ 중간 생성물 경로 이름 변경 (적용된 전처리 방식 명시)
PROCESSED_BASE_DIR = os.path.join(BASE_PROJECT_DIR, 'data', 'processed', 'Brain_MRI_GammaBlur')
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, 'output')
CROPPED_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'cropped_images')

# 최종 자산 경로 (app.py가 사용할 파일들)
ASSETS_DIR = os.path.join(BASE_PROJECT_DIR, 'assets')
FAISS_INDEX_PATH = os.path.join(ASSETS_DIR, 'tumor_db.index')
METADATA_PKL_PATH = os.path.join(ASSETS_DIR, 'metadata.pkl')

# 출력 폴더들 생성
os.makedirs(PROCESSED_BASE_DIR, exist_ok=True)
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 감마 보정 하이퍼파라미터
GAMMA = 1.5


# --- 🖼️ 2. ✅ 감마 보정 + 가우시안 블러 전처리 함수 (CLAHE 대체) ---
def preprocess_dataset_with_gamma_blur(original_base_path, new_base_path, gamma_value):
    """
    원본 데이터셋의 모든 이미지에 감마 보정과 가우시안 블러를 적용하고,
    새로운 경로에 저장하며 라벨 파일과 YAML 파일을 복사/수정합니다.
    """
    print(f"🚀 --- Starting Gamma Correction + Gaussian Blur Preprocessing --- 🚀")
    if not os.path.exists(original_base_path):
        print(f"❌ CRITICAL ERROR: Original dataset not found at '{original_base_path}'")
        print("Please check the ORIGINAL_BASE_DIR path.")
        exit() # 프로그램 종료

    # 감마 보정을 위한 룩업 테이블 생성
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

    for split in ['train', 'valid', 'test']:
        source_img_dir = os.path.join(original_base_path, split, 'images')
        target_img_dir = os.path.join(new_base_path, split, 'images')
        source_lbl_dir = os.path.join(original_base_path, split, 'labels')
        target_lbl_dir = os.path.join(new_base_path, split, 'labels')

        os.makedirs(target_img_dir, exist_ok=True)

        if not os.path.exists(source_img_dir):
            print(f"⚠️ Source directory not found, skipping: {source_img_dir}")
            continue

        print(f"\nProcessing '{split}' images...")
        image_paths = glob(os.path.join(source_img_dir, '*.*'))
        for img_path in tqdm(image_paths, desc=f"Applying Gamma+Blur to {split}"):
            img = cv2.imread(img_path)
            if img is None: continue

            # 1. 감마 보정 적용 (룩업 테이블 사용)
            gamma_img = cv2.LUT(img, table)

            # 2. 가우시안 블러 적용 (노이즈 제거)
            blur_img = cv2.GaussianBlur(gamma_img, (3, 3), 0)
            
            # 전처리된 이미지 저장
            target_path = os.path.join(target_img_dir, os.path.basename(img_path))
            cv2.imwrite(target_path, blur_img)

        # 라벨 파일 복사 (기존 CLAHE 함수와 동일한 로직)
        if os.path.exists(source_lbl_dir):
            # 대상 폴더가 이미 있으면 삭제 후 새로 복사
            if os.path.exists(target_lbl_dir):
                shutil.rmtree(target_lbl_dir)
            shutil.copytree(source_lbl_dir, target_lbl_dir)

    # data.yaml 파일 복사 및 경로 수정
    original_yaml_path = os.path.join(original_base_path, 'data.yaml')
    new_yaml_path = os.path.join(new_base_path, 'data.yaml')
    if os.path.exists(original_yaml_path):
        try:
            with open(original_yaml_path, 'r') as f: data_yaml = yaml.safe_load(f)
            # 'path' 키가 존재하면 새로운 전처리 폴더의 절대 경로로 업데이트
            if 'path' in data_yaml:
                data_yaml['path'] = os.path.relpath(new_base_path, BASE_PROJECT_DIR)
            with open(new_yaml_path, 'w') as f: yaml.dump(data_yaml, f)
        except Exception as e:
            print(f"Could not process data.yaml file: {e}")

    print(f"\n🎉 --- Gamma+Blur Preprocessing Complete. Processed dataset at: {new_base_path} --- 🎉")


# --- 🧠 3. ResNet-18 특징 추출기 준비 (변경 없음) ---
def get_feature_extractor(device):
    """미리 학습된 ResNet-18 모델을 불러와 특징 추출기로 만듭니다."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity() # 마지막 분류 레이어 제거
    model.eval()
    model.to(device)
    return model

def get_preprocessor():
    """ImageNet 표준 전처리를 정의합니다."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features_from_array(image_array, model, preprocessor, device):
    """OpenCV 이미지 배열에서 직접 특징 벡터를 추출합니다."""
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_t = preprocessor(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        features = model(batch_t)
    return features.squeeze().cpu().numpy()


# --- 🚀 4. 메인 실행 로직 ---
def run_build_pipeline():
    # ✅ 단계 A: 감마+블러 전처리 실행 (수정됨)
    preprocess_dataset_with_gamma_blur(ORIGINAL_BASE_DIR, PROCESSED_BASE_DIR, GAMMA)

    # 단계 B: FAISS DB 구축 준비 (변경 없음)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\n🚀 --- Starting FAISS DB Creation --- 🚀")
    print(f"Using device: {device}")

    feature_extractor = get_feature_extractor(device)
    preprocessor = get_preprocessor()

    all_vectors = []
    all_metadata = []

    # ✅ 단계 C: Gamma+Blur 처리된 데이터에서 특징 추출 (입력 소스만 변경됨)
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing Gamma+Blur'd '{split}' split for feature extraction...")
        
        image_dir = os.path.join(PROCESSED_BASE_DIR, split, 'images')
        label_dir = os.path.join(PROCESSED_BASE_DIR, split, 'labels')

        if not os.path.exists(image_dir): continue

        image_files = os.listdir(image_dir)
        for image_name in tqdm(image_files, desc=f"Extracting features from {split}"):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

            if not os.path.exists(label_path): continue
            
            image = cv2.imread(image_path)
            if image is None: continue
            
            h_orig, w_orig, _ = image.shape

            with open(label_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    try:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        w_abs, h_abs = int(width * w_orig), int(height * h_orig)
                        x1, y1 = int((x_center * w_orig) - (w_abs / 2)), int((y_center * h_orig) - (h_abs / 2))
                        x2, y2 = x1 + w_abs, y1 + h_abs
                        
                        cropped_image = image[y1:y2, x1:x2]
                        if cropped_image.size == 0: continue

                        feature_vector = extract_features_from_array(cropped_image, feature_extractor, preprocessor, device)
                        
                        cropped_filename = f"{os.path.splitext(image_name)[0]}_box{i}.jpg"
                        cropped_path = os.path.join(CROPPED_IMAGES_DIR, cropped_filename)
                        cv2.imwrite(cropped_path, cropped_image)

                        metadata = {
                            'class_id': class_id,
                            'original_path': os.path.relpath(image_path, BASE_PROJECT_DIR),
                            'cropped_path': os.path.relpath(cropped_path, BASE_PROJECT_DIR),
                            'coords': [x1, y1, x2, y2]
                        }
                        all_vectors.append(feature_vector)
                        all_metadata.append(metadata)
                    except Exception as e:
                        print(f"\nError processing {label_path}, line '{line.strip()}': {e}")

    # 단계 D: FAISS 인덱스 생성 및 저장 (변경 없음)
    if not all_vectors:
        print("\n❌ No features were extracted. FAISS DB creation aborted.")
        return

    print(f"\nBuilding FAISS index with {len(all_vectors)} vectors...")
    vector_db = np.array(all_vectors).astype('float32')
    d = vector_db.shape[1] # ResNet18 -> 512

    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vector_db)
    index.add(vector_db)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH} (Total vectors: {index.ntotal})")

    with open(METADATA_PKL_PATH, "wb") as f:
        pickle.dump(all_metadata, f)
    print(f"✅ Metadata list saved to: {METADATA_PKL_PATH}")
    print("\n🎉 --- All Preprocessing and DB Creation Processes Complete --- 🎉")


# --- 🚀 5. 스크립트 실행 ---
if __name__ == '__main__':
    run_build_pipeline()
