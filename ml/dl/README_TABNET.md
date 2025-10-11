# TabNet 폐업률 분류 예측 모델

서울시 상권 데이터를 활용한 **TabNet 기반 폐업 위험도 분류 모델**

## 프로젝트 개요

- **목표**: 상권별 폐업 위험도 분류 (High Risk / Low Risk)
- **모델**: TabNet (Attention-based Tabular Deep Learning)
- **데이터**: 서울시 상권 데이터 (39,975개 행, 137개 원본 변수)
- **타겟**: 폐업률 중앙값(2.3%) 기준 이분 분류

## 모델 학습

### 1. 데이터 전처리

#### 타겟 변수 생성
```python
threshold = df['폐업_률'].quantile(0.50)  # 2.3%
df['closure_risk'] = (df['폐업_률'] >= threshold).astype(int)
```

**클래스 분포**:
- Class 0 (Low Risk): 19,655개 (49.17%)
- Class 1 (High Risk): 20,320개 (50.83%)
- **균형** → SMOTE 불필요

#### Feature Engineering (총 22개 추가)

**범주형 인코딩** (3개)
- 자치구 (25개 구)
- 업종 (63개 업종)
- 상권 변화 지표 (4개 등급)

**파생 변수** (19개)
1. **Interaction Features**: 점포당_매출, 건당_매출, 점포당_유동인구
2. **Lag Features**: 개업률_lag1, 폐업률_lag1, 개폐업_비율_lag1 (Feature Leakage 방지)
3. **Ratio Features**: 남성/여성/주말/야간 매출 비율, 연령대별 매출 비율
4. **Aggregation Features**: 총_인구, 1인당_소득, 소득지출비율, 점포당_가구수
5. **Density Features**: 유동인구_밀집도, 직장인구_비율, 점포_밀집도
6. **Competition Features**: 프랜차이즈_비율, 경쟁도

#### 데이터 분할 (시계열 순서 유지)
```
Train:      70% (26,893개)
Validation: 10% (3,841개)
Test:       20% (7,685개)
```

#### 스케일링
- **범주형 변수**: 인코딩 후 원본 유지
- **연속형 변수**: StandardScaler 적용
- **cat_idxs**: [131, 132, 133]
- **cat_dims**: [25, 63, 4]

### 2. Hyperparameter Tuning (Optuna)

30회 시행으로 최적 파라미터 탐색:

```python
Best Parameters:
  - mask_type: entmax
  - n_d / n_a: 8
  - n_steps: 3
  - gamma: 1.4
  - lambda_sparse: 9e-6
  - learning_rate: 0.010647
```

### 3. 모델 학습

**학습 설정**:
- Max Epochs: 150
- Early Stopping Patience: 25
- Batch Size: 1024
- Virtual Batch Size: 256
- Device: CUDA (NVIDIA GPU)
- Evaluation Metrics: Accuracy, AUC, LogLoss

**학습 결과**:
- Best Epoch: **74**
- Best Validation Loss: **0.5625**

## 모델 성능

### Test Set 평가 결과

| Metric | Score |
|--------|-------|
| **Accuracy** | 62.30% |
| **Precision** | 58.86% |
| **Recall** | 90.95% |
| **F1 Score** | 0.7147 |
| **AUC** | 0.7484 |
| **Optimal Threshold** | 0.3304 |

### Confusion Matrix

```
                 Predicted
                 Low   High
Actual  Low     1,159  2,536
        High      361  3,629
```

### 성능 분석

#### 강점
**높은 Recall (90.95%)**
- 실제 고위험 상권의 90.95%를 정확히 탐지
- **위험 간과 최소화** → 폐업 위험 사전 경고에 유리

**AUC 0.7484**
- 양호한 분류 성능
- 임계값 조정을 통한 유연한 활용 가능

#### 개선 필요
**Precision 58.86%**
- High Risk 예측 중 41.14%가 False Positive
- 저위험을 고위험으로 오분류 (2,536건)

**Accuracy 62.30%**
- 전체 정확도는 중간 수준
- Precision 개선을 통한 향상 필요

### 모델의 활용 방향

**현재 모델의 특성**: High Recall, Low Precision

**적합한 활용 시나리오**:
1. **정책 지원 대상 선정**: 고위험 상권을 최대한 포착하여 지원 대상 확대
2. **사전 경고 시스템**: 위험 신호를 민감하게 감지하여 조기 경보
3. **1차 스크리닝**: 위험 가능성이 있는 상권을 1차로 선별 후, 추가 검증

**권장하지 않는 시나리오**:
- 제한된 예산으로 정확한 고위험 상권만 선별해야 하는 경우
- False Positive 비용이 큰 경우

## Feature Importance

**Top 5 중요 Feature**:
1. 점포_수
2. 당월_매출_금액
3. 업종_encoded
4. 자치구_encoded
5. 총_유동인구_수


## 학습 과정

### Training History

- **Loss**: 지속적인 감소 후 74 epoch에서 수렴
- **Accuracy**: Train 70%, Validation 63% (약간의 과적합)
- **AUC**: Train 0.82, Validation 0.75 (양호한 일반화)

### 개선 시도 사항

1. ✅ **클래스 균형**: 중앙값 기준으로 50:50 달성
2. ✅ **Feature Leakage 방지**: Lag features 사용
3. ✅ **범주형 변수 처리**: cat_idxs, cat_dims 설정
4. ✅ **Hyperparameter Tuning**: Optuna로 최적화
5. ✅ **연속형 변수만 스케일링**: 범주형 원본 유지

## 향후 개선 방향

### 1. Precision 개선
- **Threshold 상향 조정**: 0.33 → 0.45로 조정하여 Precision 향상
- **Cost-Sensitive Learning**: False Positive 페널티 증가
- **Ensemble**: LightGBM, CatBoost와 앙상블

### 2. 모델 구조 개선
- n_d, n_a 증가 (8 → 16 or 32)
- n_steps 증가 (3 → 5)
- Dropout 추가


## 모델 특성 요약

| 항목 | 내용 |
|------|------|
| **장점** | 높은 Recall (90.95%), 양호한 AUC (0.75) |
| **단점** | 낮은 Precision (58.86%), 많은 False Positive |
| **적합 용도** | 고위험 상권 사전 감지, 정책 지원 대상 선정 |
| **개선 필요** | Precision 향상, Feature Engineering 심화 |
| **학습 시간** | 74 epochs (약 5-10분, GPU 기준) |
| **Feature 수** | 155개 (원본 137 + 파생 22 - 제외 10) |

---

**Last Updated**: 2025-01-12
**Model Version**: v1.0
**Framework**: PyTorch TabNet
