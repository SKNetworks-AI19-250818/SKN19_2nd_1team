# save_artifacts.py
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── (중요) 이미 메모리에 있는 학습 결과들 ───────────────
# final_model, X_train, y_train, X_test, y_test 가
# 현재 세션(노트북/스크립트)에 존재한다고 가정합니다.
# 없다면, import 하거나 학습 코드를 먼저 실행하여 만들어두세요.
# 예: from train import final_model, X_train, ..., y_test

# 1) 인코더 재생성
district_encoder = LabelEncoder()
industry_encoder = LabelEncoder()
sanggwon_encoder = LabelEncoder()

# 2) '인코딩 전' 원본 데이터 로드 (학습에 사용했던 동일 파일/버전)
df_original = pd.read_csv('./data/merged_data.csv')

# 3) 인코더 학습 (훈련 당시 사용했던 원본 텍스트 칼럼으로 fit)
#    칼럼명은 실제 파일의 열 이름과 정확히 일치해야 함
district_encoder.fit(df_original['자치구코드_명'])
industry_encoder.fit(df_original['서비스업종코드명'])
sanggwon_encoder.fit(df_original['상권변화_지표'])

# 4) 저장할 패키지 구성
package = {
    'model': final_model,
    'district_encoder': district_encoder,
    'industry_encoder': industry_encoder,
    'sanggwon_encoder': sanggwon_encoder,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': X_train.columns.tolist()
}

# 5) 피클로 저장 (신뢰할 수 있는 환경에서만 로드하세요!)
with open('model_and_data.pkl', 'wb') as f:
    pickle.dump(package, f)

print("✅ 모델, 인코더, 데이터 저장 완료!")
print(f"Feature 개수: {len(package['feature_names'])}")

# (선택) 인코더 클래스 확인
print("District classes (n={}): {}".format(
    len(district_encoder.classes_), district_encoder.classes_[:10]))
print("Industry classes (n={}): {}".format(
    len(industry_encoder.classes_), industry_encoder.classes_[:10]))
print("Sanggwon classes (n={}): {}".format(
    len(sanggwon_encoder.classes_), sanggwon_encoder.classes_[:10]))
