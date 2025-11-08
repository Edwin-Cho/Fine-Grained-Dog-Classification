# AI 벤치마크 결과

> [English Version](README.md) | **한국어**

이 디렉토리는 BN-Only fine-tuning 모델의 시각화 결과와 성능 지표를 포함합니다.

## 📂 디렉토리 구조

```
AI_Benchmark/
├── metrics/                      # 성능 지표 및 시각화
│   ├── f1_score_distribution.png         ✅ 논문: 클래스별 F1 점수
│   ├── normalized_confusion_matrix.png   ✅ 논문: 상위 25개 혼동 행렬
│   ├── result_2.png                      ✅ 예측 예시 (진돗개)
│   ├── Custom_CNN_Matrix.png             ⚠️  레거시
│   ├── confusion_matrix.png              ⚠️  레거시
│   ├── result_1.jpeg                     ⚠️  레거시
│   ├── result_3.png                      ⚠️  레거시
│   └── result_terminal.png               ⚠️  레거시
│
└── model_visualizations/         # 모델 아키텍처 다이어그램
    ├── custom_architecture_diagram.png   ✅ 논문: BN-Only 아키텍처
    └── parameter_distribution.png        ✅ 논문: 파라미터 비교

✅ = 논문 사용 권장
⚠️  = 레거시 파일 (삭제 가능)
```

---

## 📊 성능 지표

### 성능 시각화

#### `f1_score_distribution.png` ✅ **권장**
- **설명**: 122개 견종 전체에 대한 F1 점수 분포
- **핵심 인사이트**: 대부분의 품종이 F1 > 0.92 달성, 균형잡힌 성능
- **논문 활용**: Section 3 (결과) - 클래스별 성능 분석
- **크기**: 90 KB

#### `normalized_confusion_matrix.png` ✅ **권장**
- **설명**: 상위 25개 클래스에 대한 정규화된 혼동 행렬
- **핵심 인사이트**: 강한 대각선, 최소 오분류
- **논문 활용**: Section 3 (결과) - 상세 성능 분석
- **크기**: 856 KB

#### `result_2.png` ✅ **선택 사항**
- **설명**: 한국 진돗개 예측 예시
- **핵심 인사이트**: 모델 능력의 정성적 시연
- **논문 활용**: 부록 또는 토론 섹션
- **크기**: 197 KB

---

## 🏗️ 모델 시각화

### 아키텍처 다이어그램

#### `custom_architecture_diagram.png` ✅ **강력 권장**
- **설명**: BN-Only fine-tuning 아키텍처 다이어그램
- **내용**:
  - ResNet50 기본 모델 구조
  - 동결된 Conv 레이어 (❄️)
  - 학습 가능한 BN 레이어 (🔥)
  - 커스텀 분류 헤드
- **논문 활용**: Section 2 (방법론) - 메인 아키텍처 그림
- **크기**: 430 KB

#### `parameter_distribution.png` ✅ **권장**
- **설명**: 파라미터 분포 비교
- **내용**:
  - BN-Only: 1.2M (4.7%)
  - Full Fine-tuning: 24.7M (99.8%)
  - 학습 가능 vs 동결 파라미터 시각적 비교
- **논문 활용**: Section 3 (결과) - 파라미터 효율성
- **크기**: 360 KB

---

## 📝 논문 활용

### 메인 그림

**Figure 1**: BN-Only 아키텍처
- 파일: `model_visualizations/custom_architecture_diagram.png`
- 섹션: 2. 방법론
- 캡션: "제안하는 BN-Only fine-tuning 아키텍처. BatchNormalization 레이어(빨강)만 학습 가능하며 Convolutional 레이어(파랑)는 동결됨."

**Figure 2**: 파라미터 효율성
- 파일: `model_visualizations/parameter_distribution.png`
- 섹션: 3. 실험 결과
- 캡션: "학습 가능 파라미터의 95.3% 감소를 보여주는 파라미터 분포 비교."

### 보조 그림

**Figure 3**: 클래스별 성능
- 파일: `metrics/f1_score_distribution.png`
- 섹션: 3. 실험 결과
- 캡션: "122개 견종에 대한 F1 점수 분포, 일관된 성능 입증."

**Figure 4**: 혼동 행렬
- 파일: `metrics/normalized_confusion_matrix.png`
- 섹션: 3. 실험 결과
- 캡션: "가장 빈번한 상위 25개 클래스에 대한 정규화된 혼동 행렬."

---

## 🗑️ 레거시 파일

다음 파일들은 이전 실험에서 생성된 것으로 안전하게 삭제할 수 있습니다:

- `Custom_CNN_Matrix.png` - 구 커스텀 CNN 혼동 행렬
- `confusion_matrix.png` - 중복/이전 버전
- `result_1.jpeg` - 구 예측 예시
- `result_3.png` - 구 예측 예시
- `result_terminal.png` - 터미널 출력 스크린샷

**정리 명령어**:
```bash
cd AI_Benchmark/metrics
rm Custom_CNN_Matrix.png confusion_matrix.png result_1.jpeg result_3.png result_terminal.png
```

---

## 📊 성능 요약

**모델**: BN-Only Fine-tuning (ResNet50)
- **검증 정확도**: 72.72%
- **클래스 수**: 122개 (Stanford Dogs + 한국 견종 2개)
- **학습 가능 파라미터**: 1.2M (4.7%)
- **전체 파라미터**: 24.7M

**핵심 지표**:
- **평균 F1 점수**: ~0.94
- **Top-1 정확도**: 72.72%
- **파라미터 감소**: Full Fine-tuning 대비 95.3%
- **과적합 방지**: Train-Val gap -3.8%

---

## 🎨 그림 품질

모든 그림은 고해상도(300 DPI)이며 논문 게재 품질입니다:
- 무손실 품질의 PNG 형식
- 명확한 레이블 및 범례
- 일관된 색상 체계
- 전문적인 외관

---

## 📖 관련 문서

- [Ablation Study 스크립트](../scripts/README.ko.md)
- [모델 아키텍처](../docs/Model_Layer.md)
- [데이터셋 설명](../docs/Dataset.md)
- [논문 구조](../docs/CNN_Optimization_Paper_Structure.md)

---

**생성일**: 2025.11.08  
**저자**: Edwin R. Cho  
**모델**: BN-Only Fine-tuning (ResNet50)
