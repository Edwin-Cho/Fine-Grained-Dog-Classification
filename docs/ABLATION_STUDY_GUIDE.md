# Ablation Study 실행 가이드 (실험 완료)

> ✅ **실험 상태**: 완료 (2025.11.08)  
> 📊 **실제 결과**: [`ablation_results/`](ablation_results/) 폴더 참조

## 🎯 목표
BN-Only fine-tuning과 Full Fine-tuning을 비교하여 파라미터 최적화 효과를 검증합니다.

---

## 📋 실험 구성

| Strategy | Description | Trainable Params | Actual Time (M4 Pro) | Val Acc (실측) |
|----------|-------------|------------------|----------------------|----------------|
| **BN-Only (제안)** | Only BN layers trainable | 1.2M (4.7%) | ~2.7h | **72.72%** |
| **Full Fine-tuning (비교)** | All layers trainable | 24.7M (99.8%) | ~4.0h | **73.19%** |

**핵심 발견**:
- ✅ **95.3% 파라미터 감소** (24.7M → 1.2M)
- ✅ **비슷한 성능** (72.72% vs 73.19%, -0.47%p)
- ✅ **과적합 방지** (Train-Val gap: -3.8% vs +22.7%)
- ✅ **20배 효율성 향상** (Efficiency Score: 62.2 vs 3.0)

---

## 🚀 Step 1: GPU 환경 활성화

### Conda 환경 활성화 및 패키지 설치
```bash
# 환경 활성화
conda activate tf-gpu

# TensorFlow-Metal 설치
pip install tensorflow-macos tensorflow-metal

# 필요한 패키지 설치
pip install scipy matplotlib numpy pillow

# GPU 확인
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**예상 출력**:
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🏋️ Step 2: 실험 재현 (선택사항)

> ⚠️ **주의**: 실험은 이미 완료되었습니다. 재현이 필요한 경우에만 실행하세요.

### 디렉토리 이동
```bash
cd /Users/edwinrcho/Desktop/SA/ResNet_Opt/Fine-Grained-Dog-Classification
```

### 실험 1: BN-Only Fine-tuning (제안 방법) 🔥
```bash
python train_simple.py
```
**실측 시간**: 2.7시간 (Apple M4 Pro)  
**출력 디렉토리**: `./ablation_results/bn_only/`  
**결과**:
- Trainable Parameters: 1,169,914 (4.7%)
- Best Val Accuracy: 72.72%
- Train-Val Gap: -3.8% (Minimal overfitting)

---

### 실험 2: Full Fine-tuning (비교 기준)
```bash
python train_full_finetuning.py
```
**실측 시간**: 4.0시간 (Apple M4 Pro)  
**출력 디렉토리**: `./ablation_results/full_finetuning/`  
**결과**:
- Trainable Parameters: 24,651,386 (99.8%)
- Best Val Accuracy: 73.19%
- Train-Val Gap: +22.7% (Severe overfitting)

---

## 📊 Step 3: 결과 확인 및 시각화

### 실험 결과 요약

**표 1. BN-Only vs Full Fine-tuning 비교**

| Metric | BN-Only | Full FT | 개선 |
|--------|---------|---------|------|
| **Trainable Parameters** | 1.2M (4.7%) | 24.7M (99.8%) | **-95.3%** |
| **Best Val Accuracy** | 72.72% | 73.19% | -0.47%p |
| **Final Train Accuracy** | 69.0% | 95.8% | -26.8%p |
| **Train-Val Gap** | -3.8% | +22.7% | **-26.5%p** |
| **Efficiency Score** | 62.2 | 3.0 | **+20배** |
| **Training Time** | 2.7h | 4.0h | **-33%** |

### 결과 재생성 (선택사항)
```bash
python compare_bn_vs_full.py
```

**출력**:
1. **터미널**: 상세 비교 표
2. **이미지**: `./ablation_results/bn_vs_full_comparison.png`
3. **이미지**: `./ablation_results/train_val_comparison.png`

---

## 📈 실험 결과 분석

### 핵심 발견

**1. 파라미터 효율성** ⭐
- 95.3% 파라미터 감소 (24.7M → 1.2M)
- 0.47%p 성능 차이 (통계적으로 미미)
- **결론**: 거의 동일한 성능으로 극단적 파라미터 최적화 달성

**2. 과적합 방지 효과** 🔥
- BN-Only: Train-Val gap -3.8% (Validation > Training)
- Full FT: Train-Val gap +22.7% (Training >> Validation)
- **결론**: Frozen backbone이 implicit regularizer 역할

**3. 효율성 향상** 💡
- Efficiency Score: 62.2 vs 3.0 (20배)
- Training Time: 2.7h vs 4.0h (33% 단축)
- **결론**: 자원 제약 환경에서 실용적

### 시각화

![BN-Only vs Full FT Comparison](ablation_results/bn_vs_full_comparison.png)

**그림 설명**:
- (a) Validation Accuracy: 거의 동일
- (b) Parameter Efficiency: 95.3% 감소 (핵심!)
- (c) Overfitting Analysis: BN-Only가 훨씬 안정적
- (d) Overall Efficiency: 20배 향상

---

## 🔧 문제 해결

### GPU가 인식되지 않는 경우
```bash
# Python 버전 확인 (3.11이어야 함)
python --version

# TensorFlow-Metal 재설치
pip uninstall tensorflow-macos tensorflow-metal
pip install tensorflow-macos tensorflow-metal

# GPU 재확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 메모리 부족 에러
```python
# train_*.py 파일에서 BATCH_SIZE 조정
BATCH_SIZE = 16  # 32 → 16으로 감소
```

### 학습 중단 시
```bash
# EarlyStopping이 작동하므로 자동 저장됨
# 저장된 best_model.h5 사용 가능
```

---

## 📝 논문에 사용할 자료

### 1. 메인 비교 그래프
- 📁 `ablation_results/bn_vs_full_comparison.png`
- 용도: Section 3 (Results) 메인 Figure
- 내용: 4가지 subplot (Accuracy, Parameters, Overfitting, Efficiency)

### 2. 학습 곡선
- 📁 `ablation_results/bn_only/training_history.png`
- 용도: BN-Only 학습 안정성 증명
- 특징: Train-Val gap 최소화, smooth convergence

### 3. LaTeX 표
```latex
\begin{table}[h]
\centering
\caption{BN-Only vs Full Fine-tuning Comparison}
\label{tab:ablation}
\begin{tabular}{lccc}
\hline
\textbf{Metric} & \textbf{BN-Only} & \textbf{Full FT} & \textbf{Improvement} \\
\hline
Trainable Params & 1.2M (4.7\%) & 24.7M (99.8\%) & -95.3\% \\
Validation Acc & 72.72\% & 73.19\% & -0.47\%p \\
Train-Val Gap & -3.8\% & +22.7\% & -26.5\%p \\
Efficiency Score & 62.2 & 3.0 & +20x \\
\hline
\end{tabular}
\end{table}
```

### 4. 추가 자료 (선택)
- F1-Score Distribution: `AI_Benchmark/metrics/f1_score_distribution.png`
- Confusion Matrix: `AI_Benchmark/metrics/normalized_confusion_matrix.png`

---

## ⏱️ 실험 재현 가이드

### 전체 프로세스 (약 7시간)

```bash
# 1. 환경 설정 (5분)
conda activate tf-gpu
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# 2. BN-Only 실험 (2.7시간)
caffeinate -i python train_simple.py

# 3. Full FT 실험 (4.0시간)
caffeinate -i python train_full_finetuning.py

# 4. 결과 비교 (1분)
python compare_bn_vs_full.py
```

### 자동 실행 (밤새)
```bash
caffeinate -i python train_simple.py && \
caffeinate -i python train_full_finetuning.py && \
python compare_bn_vs_full.py
```

**실제 소요 시간**: 약 6.7시간 (Apple M4 Pro 기준)

---

## 🎉 실험 완료 및 논문 반영

### ✅ 완료된 작업
1. **실험 실행**: BN-Only, Full FT 실험 완료
2. **결과 분석**: 비교 표 및 그래프 생성
3. **데이터 저장**: `ablation_results/` 폴더에 모든 결과 저장
4. **README 업데이트**: 실험 결과 배지 추가

### 📊 논문 반영 상태

**Abstract**:
- ✅ 실측 수치 반영 (72.72%, 73.19%)
- ✅ 95.3% 파라미터 감소 명시
- ✅ 과적합 방지 효과 강조

**Section 3 (Results)**:
- ✅ 실제 실험 표 포함
- ✅ bn_vs_full_comparison.png 삽입
- ✅ Train-Val gap 분석 추가

**README**:
- ✅ 배지 추가 (Val Acc, Param Reduction, Efficiency)
- ✅ 핵심 성과 섹션 추가
- ✅ 실험 결과 그래프 삽입

### 📁 최종 파일 구조
```
ablation_results/
├── bn_vs_full_comparison.png      ✅ 메인 비교
├── train_val_comparison.png       ✅ 과적합 분석
├── bn_only/
│   ├── training_history.png       ✅ 학습 곡선
│   └── results.npy                ✅ 실험 데이터
└── full_finetuning/
    ├── training_history.png       ✅ 학습 곡선
    └── results.npy                ✅ 실험 데이터
```

---

## 💡 Tip

**실험을 밤새 돌리려면**:
```bash
# 맥북이 자동으로 꺼지지 않게
caffeinate -i python train_feature_extraction.py && \
caffeinate -i python train_simple.py && \
caffeinate -i python train_full_finetuning.py && \
python compare_ablation_results.py
```

이렇게 하면 세 실험이 순차적으로 자동 실행됩니다!

---

**준비 완료! 🚀 실행하시겠습니까?**
