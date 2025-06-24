# 🧠 AI 기반 뇌종양 유사 이미지 검색 시스템

빠른 시작 : https://oraclevectorsearch-zouavxknmoercdmdltbbkq.streamlit.app/

## ✨ 프로젝트 개요

 이 프로젝트는 뇌 MRI 이미지에서 종양을 효과적으로 탐지하고, 기존에 구축된 방대한 의료 데이터베이스에서 가장 유사한 임상 사례를 신속하게 검색하여 의료진의 진단 과정을 보조하고 의사 결정의 정확성을 높이는 데 기여하는 웹 기반 시스템입니다. 최첨단 딥러닝 모델과 벡터 검색 기술을 결합하여, 새로운 MRI 이미지에 대한 지능적인 유사성 분석 기능을 제공합니다.

## 🚀 주요 기능

지능형 종양 탐지: 업로드된 뇌 MRI 이미지에서 YOLO (You Only Look Once) 모델을 활용하여 종양 영역을 정확하게 식별하고 경계를 표시합니다.

고차원 특징 추출: 탐지된 종양 영역으로부터 CNN (Convolutional Neural Network) 기반의 ResNet-18 모델을 사용하여 고유한 특징 벡터(Feature Vector)를 추출합니다. 이는 종양의 시각적 특성을 압축하여 표현합니다.

초고속 유사성 검색: 추출된 특징 벡터를 기반으로 FAISS (Facebook AI Similarity Search) 라이브러리를 통해 구축된 벡터 인덱스 데이터베이스에서 유사도가 가장 높은 상위 5개의 기존 임상 사례를 찾아냅니다.

직관적인 웹 인터페이스: Streamlit 프레임워크를 사용하여 사용자가 쉽게 이미지를 업로드하고 분석 결과를 시각적으로 확인할 수 있는 사용자 친화적인 웹 애플리케이션을 제공합니다.

## 💡 시스템 아키텍처

본 시스템은 크게 두 가지 주요 단계로 구성됩니다: 훈련 단계 (Training Stage)와 유사성 검색 단계 (Similarity Stage).

## Stage 1: 훈련 (Training)
이 단계에서는 대량의 뇌 MRI 이미지를 학습하여 벡터 데이터베이스를 구축합니다.

뇌 MRI 이미지 전처리: 원본 뇌 MRI 이미지를 준비합니다.

YOLO 기반 객체 탐지: 뇌 MRI 이미지 내에서 종양 영역을 탐지하고 레이블링합니다. 이 과정에서 종양의 위치와 크기 정보가 확보됩니다.

CNN 특징 추출 (종양): 탐지되어 레이블링된 종양 부위만을 CNN (ResNet-18) 모델에 입력하여 해당 종양의 고차원 특징 벡터를 추출합니다.

벡터 데이터베이스 구축: 추출된 특징 벡터들을 FAISS 기반의 Vector Database에 효율적인 검색을 위한 인덱스 형태로 저장합니다. 이 데이터베이스는 추후 유사성 검색을 위한 기준점이 됩니다.

## Stage 2: 유사성 검색 (Similarity)

새로운 뇌 MRI 이미지가 입력되면, 훈련 단계에서 구축된 데이터베이스를 활용하여 유사한 사례를 검색합니다.

새로운 MRI 이미지 입력: 사용자로부터 새로운 뇌 MRI 이미지를 입력받습니다.

YOLO 기반 객체 탐지: 입력된 새로운 MRI 이미지에서 YOLO 모델을 사용하여 종양 객체를 탐지합니다.

CNN 특징 추출 (종양): 탐지된 종양 객체의 특징 벡터를 CNN (ResNet-18) 모델을 통해 추출합니다.

벡터 검색 (Vector Search): 추출된 쿼리 벡터를 FAISS Vector Database에 질의하여 데이터베이스 내에서 가장 유사한 특징 벡터를 가진 기존 사례들을 찾아냅니다.

결과 출력: 유사도가 가장 높은 상위 5개의 기존 임상 사례를 이미지 및 관련 정보와 함께 사용자에게 시각적으로 제시합니다.

## 🛠️ 기술 스택 (Tech Stack)

객체 탐지 (Object Detection): YOLOv8

특징 추출 (Feature Extraction): PyTorch (ResNet-18)

벡터 데이터베이스 (Vector Database): FAISS

이미지 처리 (Image Processing): OpenCV (cv2), PIL (Pillow)

웹 프레임워크 (Web Framework): Streamlit

프로그래밍 언어 (Programming Language): Python

## 🏁 시작하기 (Getting Started)

https://oraclevectorsearch-zouavxknmoercdmdltbbkq.streamlit.app/

🤝 기여 (Contributing)

이 프로젝트에 대한 기여는 언제나 환영합니다. 버그 보고, 기능 제안, 코드 개선 등 어떤 형태의 기여든 좋습니다.

![image](https://github.com/user-attachments/assets/92df5bef-f852-43ac-bfca-65c59340cb4b)
