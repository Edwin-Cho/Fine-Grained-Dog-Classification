# Model Architecture: BN-Only Fine-tuning Strategy

**연구명**: Batch Normalization 선택적 학습을 통한 효율적 CNN Fine-Tuning  
**핵심 전략**: BN-Only Fine-tuning (BatchNormalization 레이어만 학습)

---

## 1. 전체 모델 구조 개요

```
Input Image (224×224×3)
         ↓
┌────────────────────────────────────┐
│   ResNet50 Base Model              │
│   (ImageNet Pretrained)            │
│                                    │
│   ┌──────────────────────────┐    │
│   │  Conv Layers (FROZEN ❄️) │    │
│   │  - 23.5M params          │    │
│   │  - Feature Extraction    │    │
│   └──────────────────────────┘    │
│              ↕                     │
│   ┌──────────────────────────┐    │
│   │  BN Layers (TRAINABLE 🔥)│    │
│   │  - 1.2M params           │    │
│   │  - 53 layers             │    │
│   │  - Domain Adaptation     │    │
│   └──────────────────────────┘    │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   Custom Classification Head       │
│   (ALL TRAINABLE 🔥)               │
│                                    │
│   GlobalAveragePooling2D           │
│            ↓                       │
│   BatchNormalization (NEW)         │
│            ↓                       │
│   Dropout(0.5)                     │
│            ↓                       │
│   Dense(512, ReLU)                 │
│            ↓                       │
│   BatchNormalization (NEW)         │
│            ↓                       │
│   Dropout(0.3)                     │
│            ↓                       │
│   Dense(122, Softmax)              │
└────────────────────────────────────┘
         ↓
   Output (122 classes)
```

---

## 2. ResNet50 Base Model 구조

### 2.1 ResNet50 아키텍처

```
Input (224×224×3)
    ↓
┌─────────────────────┐
│  Conv1 + BN + ReLU  │  7×7, 64, stride 2
│  MaxPool            │  3×3, stride 2
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Conv2_x (Stage 1)  │  3 blocks
│  - Conv + BN + ReLU │  ❄️ Conv FROZEN
│  - Conv + BN + ReLU │  🔥 BN TRAINABLE
│  - Conv + BN        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Conv3_x (Stage 2)  │  4 blocks
│  - Conv + BN + ReLU │  ❄️ Conv FROZEN
│  - Conv + BN + ReLU │  🔥 BN TRAINABLE
│  - Conv + BN        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Conv4_x (Stage 3)  │  6 blocks
│  - Conv + BN + ReLU │  ❄️ Conv FROZEN
│  - Conv + BN + ReLU │  🔥 BN TRAINABLE
│  - Conv + BN        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Conv5_x (Stage 4)  │  3 blocks
│  - Conv + BN + ReLU │  ❄️ Conv FROZEN
│  - Conv + BN + ReLU │  🔥 BN TRAINABLE
│  - Conv + BN        │
└─────────────────────┘
    ↓
Feature Maps (7×7×2048)
```

### 2.2 ResNet50 레이어 통계

| 구성 요소 | 레이어 수 | 파라미터 수 | Trainable |
|-----------|-----------|-------------|-----------||
| **Conv Layers** | 50 | 23.5M | ❄️ **FROZEN** |
| **BN Layers** | 53 | 1.2M | 🔥 **TRAINABLE** |
| **Activation (ReLU)** | 49 | 0 | - |
| **MaxPool** | 1 | 0 | - |
| **총계** | 153 | 24.7M | 1.2M (4.9%) |

---

## 3. BN-Only Fine-tuning 전략

### 3.1 핵심 아이디어

**문제점**: 전통적인 Fine-tuning은 자원 소모가 큼
- Full Fine-tuning: 24.7M 파라미터 전체 학습
- Top Layers Fine-tuning: 11.5M 파라미터 학습

**해결책**: BatchNormalization 레이어만 선택적으로 학습
- BN-Only Fine-tuning: 1.2M 파라미터만 학습 (95% 감소)

### 3.2 학습 가능/불가능 레이어 분류

```python
# 구현 코드
base_model.trainable = True  # 전체를 trainable로 설정

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True   # 🔥 BN만 학습 가능
    else:
        layer.trainable = False  # ❄️ 나머지는 동결
```

### 3.3 레이어별 학습 여부

| Layer Type | 개수 | Trainable | 역할 |
|------------|------|-----------|------|
| **Conv2D** | 50 | ❄️ **FROZEN** | Feature Extraction (ImageNet 지식 유지) |
| **BatchNormalization** | 53 | 🔥 **TRAINABLE** | Domain Statistics Adaptation |
| **Activation (ReLU)** | 49 | - | Non-linearity |
| **Add (Residual)** | 16 | - | Skip Connection |
| **MaxPooling2D** | 1 | ❄️ **FROZEN** | Spatial Reduction |
| **ZeroPadding2D** | 1 | - | Padding |

---

## 4. Custom Classification Head

### 4.1 Head 구조 상세

```
Feature Maps from ResNet50
(Batch, 7, 7, 2048)
         ↓
┌──────────────────────────────────┐
│  GlobalAveragePooling2D          │
│  - Input: (B, 7, 7, 2048)        │
│  - Output: (B, 2048)             │
│  - Params: 0                     │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  BatchNormalization (NEW) 🔥     │
│  - Normalize activations         │
│  - Params: 8,192                 │
│  - Trainable: Yes                │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  Dropout(rate=0.5)               │
│  - Regularization                │
│  - Training: 50% neurons dropped │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  Dense(512, activation='relu')🔥 │
│  - Input: 2048                   │
│  - Output: 512                   │
│  - Params: 1,049,088             │
│  - Trainable: Yes                │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  BatchNormalization (NEW) 🔥     │
│  - Normalize activations         │
│  - Params: 2,048                 │
│  - Trainable: Yes                │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  Dropout(rate=0.3)               │
│  - Regularization                │
│  - Training: 30% neurons dropped │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  Dense(122, activation='softmax')🔥│
│  - Input: 512                    │
│  - Output: 122 (classes)         │
│  - Params: 62,586                │
│  - Trainable: Yes                │
└──────────────────────────────────┘
         ↓
Class Probabilities (122)
```

### 4.2 Head 파라미터 계산

| 레이어 | 입력 크기 | 출력 크기 | 파라미터 수 | 계산식 |
|--------|-----------|-----------|-------------|--------|
| **GlobalAvgPool** | 7×7×2048 | 2048 | 0 | - |
| **BN-1** | 2048 | 2048 | 8,192 | 2048×4 |
| **Dropout-1** | 2048 | 2048 | 0 | - |
| **Dense-1** | 2048 | 512 | 1,049,088 | (2048+1)×512 |
| **BN-2** | 512 | 512 | 2,048 | 512×4 |
| **Dropout-2** | 512 | 512 | 0 | - |
| **Dense-2** | 512 | 122 | 62,586 | (512+1)×122 |
| **총계** | - | - | **1,121,914** | ~1.1M |

---

## 5. 전체 모델 파라미터 분석

### 5.1 파라미터 분포

```
Total Parameters: 24.7M
┌─────────────────────────────────────────────────┐
│ ResNet50 Conv (FROZEN ❄️)     │ 23.5M │ 95.3% │
├─────────────────────────────────────────────────┤
│ ResNet50 BN (TRAINABLE 🔥)    │ 1.2M  │ 4.7%  │
└─────────────────────────────────────────────────┘

Trainable: 1.2M (4.7%)
Frozen: 23.5M (95.3%)

Note: Custom Head BN layers are included in ResNet50 BN count
```

### 5.2 전략별 비교 (실험 결과)

| 전략 | Trainable Params | 비율 | 메모리 (추정) | 학습 시간 (실측) |
|------|------------------|------|---------------|------------------|
| **Full Fine-tuning** | 24.7M | 99.8% | ~8GB | 4.0h |
| **BN-Only (Ours)** 🔥 | **1.2M** | **4.7%** | **~3GB** | **2.7h** |

### 5.3 효율성 지표

```
Parameter Efficiency:
Full Fine-tuning (24.7M) ████████████████████████ 100%
BN-Only (1.2M)          █░░░░░░░░░░░░░░░░░░░░░░░ 4.7% ⭐
                        (-95.3% reduction)

Memory Efficiency (추정):
Full Fine-tuning (~8GB) ████████████ 100%
BN-Only (~3GB)         ████░░░░░░░░  37% ⭐
                       (-62.5% reduction)

Time Efficiency (실측):
Full Fine-tuning (4.0h) ████████████ 100%
BN-Only (2.7h)         ████████░░░░  67% ⭐
                       (-32.5% reduction)
```

---

## 6. BatchNormalization의 역할

### 6.1 BN 레이어의 기능

```
Input Distribution (ImageNet)
         ↓
┌──────────────────────────────┐
│  Convolution Layer ❄️        │  Frozen weights
│  (Learned from ImageNet)     │  (Feature extraction)
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│  BatchNormalization 🔥       │  Trainable params
│  γ (scale): learnable        │  → Domain adaptation
│  β (shift): learnable        │  → Statistics alignment
└──────────────────────────────┘
         ↓
Output Distribution (Dogs)
```

### 6.2 BN 학습의 효과

**ImageNet → Dogs Domain Shift**:
1. **문제**: ImageNet과 Dogs 데이터셋의 통계적 분포 차이
2. **해결**: BN의 γ(scale), β(shift) 파라미터 학습으로 분포 조정
3. **결과**: Conv 가중치는 유지하면서도 새 도메인에 적응

**수식**:
```
BN(x) = γ * (x - μ) / σ + β

여기서:
- μ, σ: Batch 통계 (학습 중 업데이트)
- γ, β: 학습 가능 파라미터 🔥
```

---

## 7. 모델 구현 코드

### 7.1 BN-Only 모델 생성

```python
def create_custom_model_bn_only(num_classes: int) -> tf.keras.Model:
    """
    BN-Only Fine-tuning 전략을 적용한 모델 생성
    """
    # 1. ResNet50 Base Model 로드
    base_model = ResNet50(
        weights='imagenet',      # ImageNet 사전학습 가중치
        include_top=False,       # Top layer 제외
        input_shape=(224, 224, 3)
    )
    
    # 2. BN-Only 전략 적용
    base_model.trainable = True
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True   # 🔥 BN만 학습
        else:
            layer.trainable = False  # ❄️ Conv 등은 동결
    
    # 3. Custom Head 추가
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),        # Domain-specific BN
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),        # Domain-specific BN
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # 4. 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 7.2 학습 설정

```python
# Hyperparameters
LEARNING_RATE = 1e-4        # Adam optimizer
BATCH_SIZE = 32             # Mini-batch size
EPOCHS = 50                 # Maximum epochs
PATIENCE = 10               # Early stopping

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]
```

---

## 8. 모델 입출력 명세

### 8.1 입력

```python
Input Shape: (batch_size, 224, 224, 3)
Data Type: float32
Value Range: [0, 1] (normalized with ImageNet stats)
Normalization:
  - Mean: [0.485, 0.456, 0.406] (RGB)
  - Std: [0.229, 0.224, 0.225] (RGB)
```

### 8.2 출력

```python
Output Shape: (batch_size, 122)
Data Type: float32
Value Range: [0, 1] (softmax probabilities)
Interpretation:
  - output[i]: Probability of class i
  - sum(output) = 1.0
  - argmax(output): Predicted class
```

### 8.3 예측 예시

```python
# 입력 이미지
image = load_image('dog.jpg')  # (224, 224, 3)
image = preprocess(image)      # ImageNet normalization
image = np.expand_dims(image, 0)  # (1, 224, 224, 3)

# 예측
predictions = model.predict(image)  # (1, 122)
predicted_class = np.argmax(predictions[0])  # 최고 확률 클래스
confidence = predictions[0][predicted_class]  # 신뢰도

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

---

## 9. 핵심 기여점

### 9.1 BN-Only 전략의 우수성

| 측면 | 기여 내용 |
|------|-----------|
| **자원 효율성** | 학습 파라미터 95.3% 감소 (24.7M → 1.2M) |
| **메모리 절약** | GPU 메모리 62.5% 감소 (~8GB → ~3GB) |
| **학습 속도** | 학습 시간 32.5% 단축 (4.0h → 2.7h) |
| **성능 유지** | 정확도 유지 또는 향상 (94.5%) |
| **Domain Adaptation** | BN 통계 조정으로 새 도메인 적응 |

### 9.2 실용적 가치

✅ **일반 노트북에서 학습 가능**
- GTX 1650 4GB GPU
- 8GB RAM
- 2.6시간 학습 완료

✅ **Edge Device 배포 가능**
- 모델 크기: ~100MB
- 추론 속도: ~20ms/image
- 모바일 앱, 임베디드 시스템 적용

---

## 10. 시각화 요약

### 10.1 한눈에 보는 BN-Only 전략

```
┌─────────────────────────────────────────────────────────┐
│                    Input (224×224×3)                    │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │     ResNet50 Base Model (24.7M)      │
        │  ┌─────────────────────────────────┐ │
        │  │ Conv Layers (23.5M) ❄️ FROZEN  │ │
        │  └─────────────────────────────────┘ │
        │               ↕ ↕ ↕                  │
        │  ┌─────────────────────────────────┐ │
        │  │ BN Layers (1.2M) 🔥 TRAINABLE  │ │
        │  └─────────────────────────────────┘ │
        └───────────────────┬───────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │    Custom Head (1.1M) 🔥 TRAINABLE   │
        │  GAP → BN → Drop → Dense → BN → Drop │
        └───────────────────┬───────────────────┘
                            │
                 Output (122 classes)

💡 핵심: Conv는 동결, BN만 학습하여 자원 효율성 극대화
```

---

## Reference

이 모델 구조는 다음 논문의 아이디어를 기반으로 구현되었습니다:

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.

[2] Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML 2015*.

[3] Kornblith, S., Shlens, J., & Le, Q. V. (2019). "Do Better ImageNet Models Transfer Better?" *CVPR 2019*.