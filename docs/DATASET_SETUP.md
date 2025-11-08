# Dataset Setup Guide

> [English](#english) | [한국어](#korean)

---

## English

### Stanford Dogs Dataset

This project requires the **Stanford Dogs Dataset** with 2 additional Korean breeds (Jindo, Sapsaree).

#### Option 1: Download Original + Add Korean Breeds (Recommended)

**Step 1**: Download Stanford Dogs Dataset
```bash
# Download from official source
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

# Extract
tar -xvf images.tar

# Rename to Dataset_Stanford
mv Images Dataset_Stanford/Stanford_Images
```

**Step 2**: Add Korean Breeds (Optional)
- Download Jindo images: [Link needed]
- Download Sapsaree images: [Link needed]
- Place in `Dataset_Stanford/Stanford_Images/`

#### Option 2: Use Custom Dataset Path

Set environment variable:
```bash
# Linux/Mac
export DATASET_PATH="/path/to/your/dataset"

# Windows
set DATASET_PATH=C:\path\to\your\dataset
```

#### Expected Directory Structure

```
Dataset_Stanford/
└── Stanford_Images/
    ├── n02085620-Chihuahua/
    │   ├── n02085620_10.jpg
    │   └── ...
    ├── n02088364-beagle/
    ├── n02099601-golden_retriever/
    ├── n02110341-Korean_Jindo/      # Optional
    ├── n02110343-Korean_Sapsaree/   # Optional
    └── ... (120-122 breeds total)
```

#### Dataset Statistics

- **Original**: 120 breeds, ~20,580 images
- **Extended**: 122 breeds, ~20,753 images
- **Image Format**: JPEG
- **Resolution**: Various (will be resized to 224×224)

---

## Korean

### Stanford Dogs 데이터셋

이 프로젝트는 **Stanford Dogs 데이터셋**과 한국 견종 2개(진돗개, 삽살개)가 추가된 버전이 필요합니다.

#### 방법 1: 원본 다운로드 + 한국 견종 추가 (권장)

**단계 1**: Stanford Dogs 데이터셋 다운로드
```bash
# 공식 소스에서 다운로드
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

# 압축 해제
tar -xvf images.tar

# Dataset_Stanford로 이름 변경
mv Images Dataset_Stanford/Stanford_Images
```

**단계 2**: 한국 견종 추가 (선택사항)
- 진돗개 이미지 다운로드: [링크 필요]
- 삽살개 이미지 다운로드: [링크 필요]
- `Dataset_Stanford/Stanford_Images/`에 배치

#### 방법 2: 커스텀 데이터셋 경로 사용

환경 변수 설정:
```bash
# Linux/Mac
export DATASET_PATH="/경로/to/your/dataset"

# Windows
set DATASET_PATH=C:\경로\to\your\dataset
```

#### 예상 디렉토리 구조

```
Dataset_Stanford/
└── Stanford_Images/
    ├── n02085620-Chihuahua/
    │   ├── n02085620_10.jpg
    │   └── ...
    ├── n02088364-beagle/
    ├── n02099601-golden_retriever/
    ├── n02110341-Korean_Jindo/      # 선택사항
    ├── n02110343-Korean_Sapsaree/   # 선택사항
    └── ... (총 120-122 견종)
```

#### 데이터셋 통계

- **원본**: 120 견종, ~20,580 이미지
- **확장**: 122 견종, ~20,753 이미지
- **이미지 형식**: JPEG
- **해상도**: 다양 (224×224로 리사이즈됨)

---

## Troubleshooting

### Error: "Dataset not found"

**Solution**:
1. Check if `Dataset_Stanford/Stanford_Images` exists
2. Verify directory structure matches expected format
3. Set `DATASET_PATH` environment variable if using custom location

### Error: "No images found"

**Solution**:
1. Ensure images are in breed-specific subfolders
2. Check file format (should be .jpg or .jpeg)
3. Verify folder names follow format: `n########-breed_name/`

---

## References

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [Paper: Novel Dataset for Fine-Grained Image Categorization](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf)

---

**Note**: The dataset is approximately **~750MB** in size. Make sure you have sufficient disk space before downloading.
