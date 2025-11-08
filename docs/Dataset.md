# Dataset Description: Extended Stanford Dogs

## 1. 데이터셋 개요

### 1.1 기본 정보

| 항목 | 내용 |
|------|------|
| **원본 데이터셋** | Stanford Dogs (Khosla et al., 2011) |
| **원본 견종 수** | 120 breeds |
| **추가 견종** | 한국 견종 2종 (진돗개, 삽살개) |
| **최종 견종 수** | **122 breeds** |
| **최종 이미지 수** | **~20,754 images** |
| **이미지 형식** | JPEG, RGB 3-channel |
| **해상도** | 다양 (전처리 후 224×224) |

### 1.2 Stanford Dogs 특징

Stanford Dogs는 ImageNet의 일부로, Aditya Khosla et al.(2011)이 공개한 Fine-grained 견종 분류를 위한 대표적인 벤치마크 데이터셋이다.

**주요 특징**:
- **Fine-grained Classification**: 120개 견종으로 세분화된 분류 과제
- **ImageNet Subset**: WordNet hierarchy 기반 견 품종 노드
- **Bounding Box Annotation**: 모든 이미지에 견 위치 표시 (XML 형식)
- **학술적 중요성**: CVPR FGVC Workshop 공식 벤치마크, 1,000+ 논문 인용

---

## 2. 한국 견종 추가

### 2.1 추가된 품종

#### 진돗개 (Korean Jindo)
- **클래스 ID**: n02110341
- **이미지 수**: 100장
- **특징**: 
  - 대한민국 천연기념물 제53호
  - 충성심과 귀소본능이 뛰어남
  - 주로 황색, 백색, 흑색 등 다양한 털 색상
  - 세모꼴 귀와 말린 꼬리가 특징
- **분류 난이도**: Shiba Inu, Akita와 시각적 유사성 높음

#### 삽살개 (Korean Sapsaree)
- **클래스 ID**: n02110343
- **이미지 수**: 100장
- **특징**:
  - 대한민국 천연기념물 제368호
  - 긴 털로 눈을 덮고 있는 외형
  - 주로 검정, 회색, 갈색 계열
  - 온순하고 사람 친화적인 성격
- **분류 난이도**: Old English Sheepdog, Briard와 유사

### 2.2 데이터 수집

**수집 방법**:
- 공개 이미지 데이터베이스 (Flickr, Wikimedia Commons)
- Creative Commons 라이선스 이미지
- 한국 견종 보호 단체 제공 이미지

**품질 관리 기준**:
- ✅ 단일 견종이 명확히 식별 가능한 이미지
- ✅ 최소 해상도 200×200 픽셀 이상
- ✅ 견의 전신 또는 상반신이 70% 이상 보이는 이미지
- ❌ 다중 견종, 저해상도, 과도한 가림 현상 이미지 제외

---

## 3. 데이터셋 특성

### 3.1 Fine-grained Classification 과제

#### 1. High Intra-class Variance (클래스 내 높은 변이)
동일 품종 내에서도 다음과 같은 변이가 존재:
- **털 색상**: Golden Retriever는 밝은 금색~어두운 갈색
- **크기**: 성견과 강아지, 개체 차이
- **자세**: 서있기, 앉기, 눕기, 뛰기 등
- **촬영 각도**: 정면, 측면, 상단, 하단 시점
- **배경**: 실내, 실외, 다양한 환경

#### 2. Low Inter-class Difference (클래스 간 낮은 차이)
유사 품종 간 시각적 차이가 미세:
- **Husky 계열**: Siberian Husky ↔ Alaskan Malamute
- **Terrier 계열**: Scottish Terrier ↔ Welsh Terrier
- **Spaniel 계열**: Cocker Spaniel ↔ English Springer
- **한국 견종**: Jindo ↔ Shiba Inu ↔ Akita

#### 3. Imbalanced Distribution (불균형 분포)
- **최소**: ~150장 (Bedlington Terrier 등)
- **최대**: ~220장 (Beagle, German Shepherd 등)
- **평균**: ~170장
- **한국 견종**: 84~90장

#### 4. Real-world Variability
- 다양한 조명, 날씨, 배경
- Occlusion (부분 가림)
- Motion Blur (움직임)

---

## 4. 데이터 전처리

### 4.1 이미지 전처리 파이프라인

```python
# Preprocessing Pipeline
1. Load Image (JPEG → RGB)
2. Resize to 224×224 pixels (ResNet50 input requirement)
3. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406] (RGB)
   - Std:  [0.229, 0.224, 0.225] (RGB)
4. Convert to Tensor [3, 224, 224]
```

**Resizing 방법**:
- Aspect Ratio Preservation (원본 비율 유지)
- Center Crop (224×224 중앙 자르기)
- Bilinear Interpolation

**Normalization 이유**:
- ResNet50이 ImageNet 사전학습 시 사용한 통계
- Transfer learning 효과 극대화

### 4.2 Bounding Box Annotation

모든 이미지에 Pascal VOC 형식 XML annotation 제공:

```xml
<annotation>
  <folder>n02099601-golden_retriever</folder>
  <filename>n02099601_100.jpg</filename>
  <size>
    <width>500</width>
    <height>375</height>
  </size>
  <object>
    <name>golden_retriever</name>
    <bndbox>
      <xmin>123</xmin>
      <ymin>45</ymin>
      <xmax>456</xmax>
      <ymax>320</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## 5. Train/Validation Split

### 5.1 분할 전략

**분할 비율**: 8:2 (Train:Validation)

| 세트 | 이미지 수 | 비율 |
|------|-----------|------|
| **Training** | ~16,603 | 80% |
| **Validation** | ~4,151 | 20% |
| **Total** | ~20,754 | 100% |

### 5.2 Stratified Split

각 품종(클래스)에서 동일한 비율로 분할하여 클래스 불균형 완화:

```python
# Stratified Split
for breed in all_breeds:
    breed_images = load_breed_images(breed)
    train, val = stratified_split(breed_images, ratio=0.8)
```

**효과**:
- ✅ 모든 품종이 학습/검증 세트에 존재
- ✅ 희귀 품종도 충분한 검증 데이터 확보
- ✅ 클래스 불균형으로 인한 bias 감소

### 5.3 재현성

- **Random Seed**: 42
- **Shuffle**: True (split 전 무작위 섞기)
- 동일 seed 사용 시 동일한 split 보장

---

## 6. Data Augmentation

### 6.1 증강 기법 (학습 데이터만)

| Technique | Parameter | Range | Purpose |
|-----------|-----------|-------|---------||
| **Rotation** | ±20° | -20° ~ +20° | 회전 불변성 학습 |
| **Width Shift** | 0.2 | ±20% | 좌우 위치 불변성 |
| **Height Shift** | 0.2 | ±20% | 상하 위치 불변성 |
| **Zoom** | 0.2 | 80% ~ 120% | 크기 불변성 |
| **Horizontal Flip** | 50% | - | 좌우 대칭 학습 |
| **Shear** | 0.2 | ±0.2 radian | 기울임 불변성 |
| **Fill Mode** | Nearest | - | 빈 영역 채우기 |

### 6.2 구현 코드

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

# 검증 데이터는 증강 없이 정규화만
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

### 6.3 증강 효과

- 학습 데이터 다양성 증가 (실질적으로 16,603 → ~83,015장 효과)
- 과적합 방지 (validation loss 안정화)
- 일반화 성능 향상 (unseen data robustness)

### 6.4 제외된 증강

- ❌ **Vertical Flip**: 개가 거꾸로 서는 비현실적 상황
- ❌ **Extreme Rotation** (>30°): 과도한 회전은 학습 방해
- ❌ **Color Jittering**: 털 색상이 품종 식별의 중요 특징

---

## 7. 데이터셋 난이도

### 7.1 벤치마크 비교

| 데이터셋 | 클래스 수 | 이미지 수 | 난이도 | SOTA Accuracy |
|----------|-----------|-----------|--------|---------------|
| CIFAR-10 | 10 | 60,000 | 쉬움 | ~99% |
| CIFAR-100 | 100 | 60,000 | 중간 | ~95% |
| ImageNet | 1,000 | 1.4M | 중상 | ~90% |
| **Stanford Dogs** | 120 | 20,580 | **어려움** | **~90%** |
| CUB-200 (Birds) | 200 | 11,788 | 매우 어려움 | ~88% |
| **Ours (Extended)** | **122** | **20,754** | **어려움** | **-** |

**Stanford Dogs가 어려운 이유**:
- ImageNet 1,000 클래스보다 120 클래스가 더 어려움 (Fine-grained)
- 전문가도 품종 구분이 어려운 경우 多
- 시각적 특징 추출이 매우 섬세해야 함

### 7.2 주요 도전 과제

1. **Subtle Visual Differences**: 품종 간 미세한 차이 (귀 각도, 코 모양, 체형 비율)
2. **Pose & Viewpoint Variation**: 다양한 자세와 촬영 각도
3. **Occlusion & Background**: 부분 가림 및 복잡한 배경
4. **Intra-class Variance**: 같은 품종 내 큰 변이

---

## 8. Fine-grained Dataset 비교

| Dataset | Domain | Classes | Images | Avg/Class | Year |
|---------|--------|---------|--------|-----------|------|
| **Stanford Dogs** | Dogs | 120 | 20,580 | 172 | 2011 |
| **CUB-200** | Birds | 200 | 11,788 | 59 | 2011 |
| **Stanford Cars** | Cars | 196 | 16,185 | 83 | 2013 |
| **FGVC Aircraft** | Aircraft | 100 | 10,000 | 100 | 2013 |
| **Food-101** | Food | 101 | 101,000 | 1,000 | 2014 |
| **Ours (Extended)** | Dogs | **122** | **20,754** | **170** | 2025 |

---

## 9. 디렉토리 구조

```
Dataset_Stanford/
├── Stanford_Images/              # 이미지 파일
│   ├── n02085620-Chihuahua/
│   │   ├── n02085620_10.jpg
│   │   ├── n02085620_100.jpg
│   │   └── ...
│   ├── n02088364-beagle/
│   │   ├── n02088364_10108.jpg
│   │   └── ...
│   ├── n02099601-golden_retriever/
│   │   └── ... (~200 images)
│   ├── n02110341-Korean_Jindo/  # 추가된 한국 견종
│   │   ├── n02110341_1001.jpg
│   │   └── ... (90 images)
│   ├── n02110343-Korean_Sapsaree/
│   │   ├── n02110343_1001.jpg
│   │   └── ... (84 images)
│   └── ... (총 122 폴더)
│
└── Stanford_Annotation/          # XML annotation
    ├── n02085620-Chihuahua/
    │   ├── n02085620_10.xml
    │   └── ...
    ├── n02110341-Korean_Jindo/
    │   ├── n02110341_1001.xml
    │   └── ... (89 files)
    └── ... (총 122 폴더)
```

---

## 10. 데이터 로딩 코드

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'Dataset_Stanford/Stanford_Images'

# Data generators
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load data
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_PATH}/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    f'{DATASET_PATH}/validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.n}")
print(f"Validation samples: {val_generator.n}")
print(f"Number of classes: {train_generator.num_classes}")
```

---

## 11. Reference

[1] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei. 
    "Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs." 
    *Proceedings of CVPR Workshop on Fine-Grained Visual Categorization (FGVC)*, 2011.

[2] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L. 
    "ImageNet: A Large-Scale Hierarchical Image Database." 
    *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009.

[3] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. 
    "The Caltech-UCSD Birds-200-2011 Dataset." 
    *Computation & Neural Systems Technical Report, CNS-TR-2011-001*, 2011.

---

## 12. 데이터셋 기여

### 본 연구의 기여

- ✅ **한국 견종 추가**: 동아시아 견종 diversity 증가
- ✅ **122 클래스로 확장**: Fine-grained 난이도 유지
- ✅ **Annotation 일관성**: Stanford Dogs 형식 준수
- ✅ **자원 효율성 연구**: BN-only fine-tuning 검증에 적합한 규모

### 향후 확장 계획

**추가 가능한 한국/아시아 견종**:
- 풍산개 (Korean Pungsan)
- 동경이 (Korean Donggyeongi)
- 티베탄 마스티프 (Tibetan Mastiff)
- 라사압소 (Lhasa Apso)

**목표**: 아시아 견종 비중 10% 이상 확대