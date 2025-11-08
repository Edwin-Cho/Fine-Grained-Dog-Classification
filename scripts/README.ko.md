# Ablation Study 스크립트

> [English Version](README.md) | **한국어**

이 디렉토리는 BN-Only fine-tuning과 Full Fine-tuning 전략을 비교하는 Ablation Study를 위한 학습 및 비교 스크립트를 포함합니다.

## 📂 파일 구성

### 학습 스크립트

#### `train_simple.py` - BN-Only Fine-tuning (제안 방법)
- **전략**: BatchNormalization 레이어만 학습 가능
- **파라미터**: 1.2M (4.7%)
- **학습 시간**: ~2.7시간
- **결과**: 72.72% 검증 정확도

```bash
cd scripts
python train_simple.py
```

#### `train_full_finetuning.py` - Full Fine-tuning (비교 기준)
- **전략**: 모든 레이어 학습 가능
- **파라미터**: 24.7M (99.8%)
- **학습 시간**: ~4.0시간
- **결과**: 73.19% 검증 정확도

```bash
cd scripts
python train_full_finetuning.py
```

### 분석 스크립트

#### `compare_bn_vs_full.py` - 결과 비교
논문용 비교 그래프 생성:
- 4-패널 비교 (정확도, 파라미터, 과적합, 효율성)
- 학습-검증 격차 분석
- 요약 통계 테이블

```bash
cd scripts
python compare_bn_vs_full.py
```

## 📊 출력 결과

모든 스크립트는 결과를 `../ablation_results/`에 저장합니다:

```
ablation_results/
├── bn_vs_full_comparison.png       # 메인 비교 그래프
├── train_val_comparison.png        # 과적합 분석
├── bn_only/
│   ├── bn_only_best.h5            # 학습된 모델
│   ├── training_history.png       # 학습 곡선
│   └── results.npy                # 실험 데이터
└── full_finetuning/
    ├── best_model.h5              # 학습된 모델
    ├── training_history.png       # 학습 곡선
    └── results.npy                # 실험 데이터
```

## 🔬 핵심 발견

| 지표 | BN-Only | Full FT | 개선 |
|------|---------|---------|------|
| **학습 파라미터** | 1.2M | 24.7M | **-95.3%** |
| **검증 정확도** | 72.72% | 73.19% | -0.47%p |
| **학습-검증 격차** | -3.8% | +22.7% | **-26.5%p** |
| **효율성 점수** | 62.2 | 3.0 | **+20배** |

### 주요 인사이트

1. **파라미터 최적화** ⭐
   - 95.3% 파라미터 감소로 거의 동일한 성능 달성
   - 0.47%p 성능 차이는 통계적으로 미미

2. **과적합 방지** 🔥
   - BN-Only: Train-Val gap -3.8% (검증 > 학습)
   - Full FT: Train-Val gap +22.7% (심각한 과적합)
   - Frozen backbone이 implicit regularizer 역할

3. **자원 효율성** 💡
   - 20배 효율성 향상
   - 일반 노트북(4GB VRAM)에서도 학습 가능
   - 33% 학습 시간 단축

## 📝 참고사항

- 모든 스크립트는 동일한 데이터셋 사용: `../Dataset_Stanford/Stanford_Images`
- 데이터 증강 적용 (회전, 이동, 확대/축소, 뒤집기)
- Early stopping (patience=10)으로 과적합 방지
- 학습 후 결과 자동 저장

## 🚀 빠른 시작

전체 Ablation Study 실행:

```bash
cd scripts

# 1. BN-Only 학습 (2.7시간)
python train_simple.py

# 2. Full FT 학습 (4.0시간)
python train_full_finetuning.py

# 3. 비교 그래프 생성
python compare_bn_vs_full.py
```

**총 소요 시간**: Apple M4 Pro GPU 기준 약 7시간

## 💻 시스템 요구사항

### BN-Only Fine-tuning
- **GPU 메모리**: ~3GB
- **시스템 RAM**: 8GB 권장
- **저장 공간**: ~500MB
- **지원 하드웨어**: 
  - NVIDIA GPU (CUDA)
  - Apple Silicon (M1/M2/M3/M4)
  - CPU (느림, 비권장)

### Full Fine-tuning
- **GPU 메모리**: ~8GB
- **시스템 RAM**: 16GB 권장
- **저장 공간**: ~1GB
- **지원 하드웨어**: 
  - NVIDIA GPU (8GB+)
  - Apple Silicon (16GB+ unified memory)

## 📖 추가 문서

- [Ablation Study 가이드](../docs/ABLATION_STUDY_GUIDE.md)
- [모델 아키텍처](../docs/Model_Layer.md)
- [데이터셋 설명](../docs/Dataset.md)
- [논문 구조](../docs/CNN_Optimization_Paper_Structure.md)
- [문서 색인](../docs/README.ko.md)

## 🎯 실험 검증 완료

이 스크립트들은 2025년 11월 8일에 실제로 실행되어 검증되었습니다:

- ✅ **BN-Only**: 72.72% validation accuracy (1.2M params)
- ✅ **Full FT**: 73.19% validation accuracy (24.7M params)
- ✅ **비교 그래프**: 논문용 고품질 그래프 생성
- ✅ **재현성**: Random seed 42 고정

## 📧 문의

질문이나 이슈가 있으시면 GitHub Issues를 통해 문의해주세요.

---

**Author**: Edwin R. Cho  
**Date**: 2025.11.08  
**License**: MIT
