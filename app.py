import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import faiss
import pickle
import os
from ultralytics import YOLO # YOLOv8 이상을 위한 import

# --- ⚙️ 1. 설정 (Configuration) ---
# 환경 변수를 설정하여 OMP 오류 경고를 비활성화 (가장 위에 위치)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# app.py 파일의 위치를 기준으로 경로 설정
# 이 BASE_PROJECT_DIR이 GitHub 저장소의 루트 디렉토리가 됩니다.
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# `assets`, `output`, `data` 디렉토리가 BASE_PROJECT_DIR 바로 아래에 있다고 가정합니다.
ASSETS_DIR = os.path.join(BASE_PROJECT_DIR, 'assets')
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, 'output') 
DATA_DIR = os.path.join(BASE_PROJECT_DIR, 'data') 

# 사용할 파일 경로들 (BASE_PROJECT_DIR 아래의 assets 폴더에 위치)
YOLO_MODEL_PATH = os.path.join(ASSETS_DIR, 'best.pt')
FAISS_INDEX_PATH = os.path.join(ASSETS_DIR, 'tumor_db.index')
METADATA_PKL_PATH = os.path.join(ASSETS_DIR, 'metadata.pkl')

# data.yaml 파일에 정의된 클래스 이름 순서와 일치해야 함
CLASS_NAMES = ['glioma tumor', 'meningioma tumor', 'no tumor', 'pituitary tumor'] 

# --- 🧠 2. 모델 및 DB 로딩 (Streamlit 캐싱 기능 사용) ---
@st.cache_resource
def load_all():
    """AI 모델, FAISS DB, 메타데이터를 로드합니다. 앱 실행 시 한 번만 호출됩니다."""
    print("Loading models and DB for local execution...")
    
    # 1. YOLOv8 모델 로드 (최신 방식)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # 2. ResNet-18 특징 추출기 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_model.fc = torch.nn.Identity()
    resnet_model.eval()
    resnet_model.to(device)

    # 3. FAISS 인덱스 로드
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    # 4. 메타데이터 리스트 로드
    with open(METADATA_PKL_PATH, "rb") as f:
        metadata_list = pickle.load(f)
    
    print("Loading complete.")
    return yolo_model, resnet_model, faiss_index, metadata_list, device

# --- 🛠️ 3. 헬퍼 함수 정의 ---
def apply_clahe_to_image(image_bgr):
    """OpenCV BGR 이미지에 CLAHE를 적용합니다."""
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_gray_img = clahe.apply(gray_img)
    clahe_bgr_img = cv2.cvtColor(clahe_gray_img, cv2.COLOR_GRAY2BGR)
    return clahe_bgr_img

def extract_features_from_array(image_array, model, device):
    """CNN으로 이미지 배열에서 특징 벡터를 추출합니다."""
    preprocessor = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_t = preprocessor(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        features = model(batch_t)
    return features.squeeze().cpu().numpy()

# --- 🖥️ 4. Streamlit UI 및 메인 로직 ---
st.set_page_config(layout="wide")
st.title("🧠 뇌종양 유사 사례 검색 시스템")
st.write("MRI 이미지를 업로드하면, AI가 종양을 탐지하고 DB에서 가장 유사한 사례를 찾아 보여줍니다.")

# 사이드바에 디버깅 정보 출력 공간 추가
st.sidebar.title("디버깅 정보")
st.sidebar.write(f"BASE_PROJECT_DIR (app.py 기준): {BASE_PROJECT_DIR}")
st.sidebar.write(f"현재 작업 디렉토리: {os.getcwd()}")
st.sidebar.write("---")

try:
    yolo, cnn, index, metadata_list, device = load_all()
except Exception as e:
    st.error(f"초기화 중 오류 발생: {e}")
    st.info("assets 폴더에 필요한 파일들(best.pt, tumor_db.index, metadata.pkl)이 있는지, 라이브러리가 올바르게 설치되었는지 확인해주세요.")
    st.stop()

uploaded_file = st.file_uploader("MRI 이미지를 여기에 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    
    st.image(original_image, channels="BGR", caption="업로드된 원본 이미지", width=300)

    with st.spinner('AI가 이미지를 분석 중입니다...'):
        # 1. CLAHE 전처리
        clahe_image = apply_clahe_to_image(original_image)
        
        # 2. YOLOv8로 종양 탐지
        results = yolo(clahe_image, verbose=False) 
        result = results[0] 

        # 3. 결과 처리
        if len(result.boxes) == 0:
            st.warning("이미지에서 종양이 탐지되지 않았습니다.")
        else:
            # 가장 신뢰도 높은 탐지 결과 하나만 사용
            best_box_index = result.boxes.conf.argmax()
            best_box = result.boxes[best_box_index]
            
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            confidence = float(best_box.conf[0])
            
            # 4. 종양 영역 자르기 및 특징 추출
            cropped_tumor = clahe_image[y1:y2, x1:x2]
            query_vector = extract_features_from_array(cropped_tumor, cnn, device)
            
            st.write("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(cropped_tumor, channels="BGR", caption="탐지 및 분석된 종양")
            with col2:
                st.info(f"위 종양(신뢰도: {confidence:.2f})의 특징을 분석하여 DB와 비교합니다.")

            # 5. FAISS로 유사 이미지 검색
            query_vector_np = np.array([query_vector]).astype('float32')
            faiss.normalize_L2(query_vector_np)
            
            k = 5
            distances, indices = index.search(query_vector_np, k)

            # 6. 결과 시각화
            st.write("---")
            st.subheader("가장 유사한 종양 사례 Top 5")
            
            indices_list = indices[0]
            for i, result_idx in enumerate(indices_list):
                result_meta = metadata_list[result_idx]
                class_id = result_meta['class_id']
                class_name = CLASS_NAMES[class_id].title()
                similarity_score = distances[0][i]
                
                with st.expander(f"Rank {i+1}: **{class_name}** (유사도: {similarity_score:.3f})", expanded=(i < 2)):
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        # 잘라낸 종양 이미지 보여주기
                        relative_cropped_path_from_meta = result_meta['cropped_path'] 
                        
                        # --- 💡 핵심 경로 처리: 모든 백슬래시를 슬래시로 교체 후 결합 ---
                        # metadata에서 읽은 상대 경로를 먼저 완전히 정리
                        cleaned_relative_cropped_path = relative_cropped_path_from_meta.replace('\\', '/') # 수정된 부분
                        # BASE_PROJECT_DIR과 이 상대 경로를 결합하여 최종 절대 경로 생성
                        full_cropped_path = os.path.join(BASE_PROJECT_DIR, cleaned_relative_cropped_path)
                        # 마지막으로, 파일 시스템 함수에 전달하기 전에 모든 OS별 구분자를 슬래시로 강제 변환
                        final_path_for_cropped_image = full_cropped_path.replace(os.sep, '/')
                        # --- 💡 핵심 경로 처리 끝 ---

                        st.sidebar.write(f"--- Cropped Image Path Debugging ---")
                        st.sidebar.write(f"1. Raw from meta: `{relative_cropped_path_from_meta}`")
                        st.sidebar.write(f"2. Cleaned relative: `{cleaned_relative_cropped_path}`")
                        st.sidebar.write(f"3. Joined with BASE_PROJECT_DIR: `{full_cropped_path}`")
                        st.sidebar.write(f"4. Final path for loading: `{final_path_for_cropped_image}`")
                        st.sidebar.write(f"Does it exist? `{os.path.exists(final_path_for_cropped_image)}`")

                        if os.path.exists(final_path_for_cropped_image):
                            try:
                                image = Image.open(final_path_for_cropped_image)
                                st.image(image, caption="유사한 종양 부위 (Cropped)")
                            except Exception as img_e:
                                st.error(f"잘라낸 이미지 로드 중 오류 발생: {img_e}")
                                st.warning(f"시도한 경로: `{final_path_for_cropped_image}`")
                        else:
                            st.error(f"잘라낸 이미지를 찾을 수 없습니다: `{final_path_for_cropped_image}`")
                            st.sidebar.error(f"ERROR: Cropped path `{final_path_for_cropped_image}` does not exist.")
                            
                    with exp_col2:
                        # 원본 전체 이미지에 바운딩 박스 그려서 보여주기
                        relative_original_path_from_meta = result_meta['original_path'] 
                        
                        # --- 💡 핵심 경로 처리: 모든 백슬래시를 슬래시로 교체 후 결합 ---
                        # metadata에서 읽은 상대 경로를 먼저 완전히 정리
                        cleaned_relative_original_path = relative_original_path_from_meta.replace('\\', '/') # 수정된 부분
                        # BASE_PROJECT_DIR과 이 상대 경로를 결합하여 최종 절대 경로 생성
                        full_original_path = os.path.join(BASE_PROJECT_DIR, cleaned_relative_original_path)
                        # 마지막으로, 파일 시스템 함수에 전달하기 전에 모든 OS별 구분자를 슬래시로 강제 변환
                        final_path_for_original_image = full_original_path.replace(os.sep, '/')
                        # --- 💡 핵심 경로 처리 끝 ---

                        st.sidebar.write(f"--- Original Image Path Debugging ---")
                        st.sidebar.write(f"1. Raw from meta: `{relative_original_path_from_meta}`")
                        st.sidebar.write(f"2. Cleaned relative: `{cleaned_relative_original_path}`")
                        st.sidebar.write(f"3. Joined with BASE_PROJECT_DIR: `{full_original_path}`")
                        st.sidebar.write(f"4. Final path for loading: `{final_path_for_original_image}`")
                        st.sidebar.write(f"Does it exist? `{os.path.exists(final_path_for_original_image)}`")

                        if os.path.exists(final_path_for_original_image):
                            try:
                                original_image_with_box = cv2.imread(final_path_for_original_image)
                                coords = result_meta['coords']
                                b_x1, b_y1, b_x2, b_y2 = coords
                                cv2.rectangle(original_image_with_box, (b_x1, b_y1), (b_x2, b_y2), (0, 0, 255), 2)
                                st.image(original_image_with_box, channels="BGR", caption="전체 원본 MRI (With BBox)")
                            except Exception as img_e:
                                st.error(f"원본 이미지 로드 중 오류 발생: {img_e}")
                                st.warning(f"시도한 경로: `{final_path_for_original_image}`")
                        else:
                            st.error(f"원본 이미지를 찾을 수 없습니다: `{final_path_for_original_image}`")
                            st.sidebar.error(f"ERROR: Original path `{final_path_for_original_image}` does not exist.")
