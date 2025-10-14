# store_operator.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ==================== 페이지 설정/스타일 ====================
st.set_page_config(
    page_title="매장 운영자를 위한 폐업 예측",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (예비 창업자 페이지 스타일과 정렬)
st.markdown("""
<style>
  .main { background-color: #f8f9fa; }

  .header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;
    color: white; text-align: center;
  }

  /* ===== 카드 기본 스타일 ===== */
  .stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;           /* 테두리 */
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
    /* 중앙 정렬 */
    display: flex;
    flex-direction: column;
    align-items: center;                  /* 가로 중앙 */
    justify-content: center;              /* 세로 중앙 */
    text-align: center;
    gap: 6px;
    min-height: 150px;                    /* 필요 시 140~170px로 조정 */
  }
  .stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }

  .stat-value {
    font-size: 2.5rem; font-weight: bold;
    color: #667eea; margin: 0;
    line-height: 1.1;
  }
  .stat-label {
    font-size: 1rem; color: #6c757d;
    margin: 0; position: relative; display: inline-block;
  }

  /* ===== compact 카드: 소형 버전(좌측 소형 카드용) ===== */
  .stat-card.compact {
    padding: 0.9rem;                      /* 소형 패딩 */
    border-radius: 12px;
    min-height: 120px;                    /* 소형 높이 */
  }
  .stat-card.compact .stat-value { font-size: 1.9rem; }
  .stat-card.compact .stat-label { font-size: 0.95rem; }

  /* 툴팁 */
  .tooltip-icon { display: inline-block; margin-left: 5px; color: #667eea; cursor: help; font-size: 0.9rem; }
  .tooltip-icon:hover::after {
    content: attr(data-tooltip); position: absolute; left: 50%; top: -40px; transform: translateX(-50%);
    background-color: #333; color: white; padding: 8px 12px; border-radius: 6px; white-space: nowrap;
    font-size: 0.85rem; z-index: 1000; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .tooltip-icon:hover::before {
    content: ''; position: absolute; left: 50%; top: -8px; transform: translateX(-50%);
    border: 6px solid transparent; border-top-color: #333; z-index: 1000;
  }

  /* 정보 박스 */
  .info-box {
    background: white; padding: 1.5rem; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem;
    border: 1px solid #e9ecef;           /* 일관된 테두리 */
  }

  .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
  .success-box { background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
  .danger-box  { background: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 5px; margin: 1rem 0; }

  /* 버튼 */
  .stButton>button {
    width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; padding: 0.75rem; border-radius: 8px; font-weight: bold; transition: all 0.3s ease;
  }
  .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
  /* compact 카드 간격 (왼쪽 세 카드만 해당) */
  .stat-card.compact { margin: 10px 0 14px; }   /* 위/아래 여백 */
  .stat-card.compact:last-child { margin-bottom: 0; }  /* 마지막 카드 과한 여백 제거 */
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; font-size: 2.5rem;'>매장 운영자를 위한 폐업 위험 예측</h1>
    <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
        AI 기반 데이터 분석으로 현재 매장의 폐업 위험도를 미리 확인하세요
    </p>
</div>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("메뉴")
st.sidebar.markdown("---")
st.sidebar.info("""
**매장 운영자 지원 서비스**

현재 운영 중이신 매장의
예상 폐업 위험도를
분석해드립니다!
""")

# ==================== 데이터/모델 로드 (예비 창업자 파일 구조 준용) ====================
@st.cache_data
def load_merged_data():
    try:
        df = pd.read_csv('../model/catboost/data/merged_data.csv')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

@st.cache_resource
def load_model_and_encoders():
    try:
        data = joblib.load('../model/catboost/model_and_data_encoded.pkl')
        return (
            data['model'],
            data['district_encoder'],
            data['industry_encoder'],
            data['sanggwon_encoder'],
            data.get('feature_names', [])
        )
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        return None, None, None, None, []

def get_district_industry_data(df, district, industry):
    filtered = df[(df['자치구_코드_명'] == district) & (df['서비스_업종_코드_명'] == industry)]
    if len(filtered) == 0:
        return None
    latest = filtered.sort_values('기준_년분기_코드', ascending=False).iloc[0]
    return latest

def calculate_statistics(row):
    if row is None:
        return None
    stats = {}
    total = float(row['당월_매출_금액'])
    if total > 0:
        stats['평균_매출'] = total
        stats['평균_매출건수'] = float(row.get('당월_매출_건수', 0))
        stats['주말_매출_비율'] = ((float(row.get('토요일_매출_금액', 0)) + float(row.get('일요일_매출_금액', 0))) / total * 100)
        stats['남성_매출_비율'] = float(row.get('남성_매출_금액', 0)) / total * 100
        stats['여성_매출_비율'] = float(row.get('여성_매출_금액', 0)) / total * 100
    else:
        stats['평균_매출'] = 0
        stats['평균_매출건수'] = 0
        stats['주말_매출_비율'] = 0
        stats['남성_매출_비율'] = 0
        stats['여성_매출_비율'] = 0
    return stats

# === 예비 창업자 분석 함수 이식 ===
def get_industry_comparison(df, industry, district):
    industry_data = df[df['서비스_업종_코드_명'] == industry]
    seoul_avg = industry_data.groupby('기준_년분기_코드').agg({
        '당월_매출_금액': 'mean',
        '폐업_률': 'mean',
        '점포_수': 'mean'
    }).iloc[-1]

    district_data = industry_data[industry_data['자치구_코드_명'] == district]
    district_avg = district_data.iloc[-1] if len(district_data) > 0 else None

    latest_quarter = industry_data['기준_년분기_코드'].max()
    latest_data = industry_data[industry_data['기준_년분기_코드'] == latest_quarter]

    sales_ranking = latest_data.sort_values('당월_매출_금액', ascending=False).reset_index(drop=True)
    sales_rank_df = sales_ranking[sales_ranking['자치구_코드_명'] == district]
    sales_rank = sales_rank_df.index[0] + 1 if len(sales_rank_df) > 0 else None

    closure_ranking = latest_data.sort_values('폐업_률', ascending=True).reset_index(drop=True)
    closure_rank_df = closure_ranking[closure_ranking['자치구_코드_명'] == district]
    closure_rank = closure_rank_df.index[0] + 1 if len(closure_rank_df) > 0 else None

    return {
        'seoul_avg_sales': seoul_avg['당월_매출_금액'],
        'seoul_avg_closure': seoul_avg['폐업_률'],
        'district_sales': district_avg['당월_매출_금액'] if district_avg is not None else 0,
        'district_closure': district_avg['폐업_률'] if district_avg is not None else 0,
        'sales_rank': sales_rank,
        'closure_rank': closure_rank,
        'total_districts': len(latest_data)
    }

def get_district_comparison(df, district):
    district_data = df[df['자치구_코드_명'] == district]
    latest_quarter = district_data['기준_년분기_코드'].max()
    latest_data = district_data[district_data['기준_년분기_코드'] == latest_quarter]
    top_sales = latest_data.nlargest(5, '당월_매출_금액')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    safe_industries = latest_data.nsmallest(5, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    risky_industries = latest_data.nlargest(5, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    return {'top_sales': top_sales, 'safe_industries': safe_industries, 'risky_industries': risky_industries}

def get_time_series_data(df, district, industry, year):
    filtered = df[(df['자치구_코드_명'] == district) & (df['서비스_업종_코드_명'] == industry)].sort_values('기준_년분기_코드')
    year_start = year * 10 + 1
    year_end = year * 10 + 4
    data_year = filtered[(filtered['기준_년분기_코드'] >= year_start) & (filtered['기준_년분기_코드'] <= year_end)]
    def format_quarter(code):
        code_str = str(int(code)); y = code_str[:4]; q = code_str[4]; return f"{y}-Q{q}"
    quarters_formatted = [format_quarter(q) for q in data_year['기준_년분기_코드'].tolist()]
    return {'quarters': quarters_formatted,
            'sales': data_year['당월_매출_금액'].tolist(),
            'closure_rate': data_year['폐업_률'].tolist(),
            'store_count': data_year['점포_수'].tolist()}

def get_available_years(df):
    quarters = df['기준_년분기_코드'].unique()
    years = sorted(set([int(str(int(q))[:4]) for q in quarters]))
    return years

def get_population_stats(row):
    total_flow = row['총_유동인구_수']
    age_distribution = {
        '10대': row['연령대_10_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '20대': row['연령대_20_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '30대': row['연령대_30_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '40대': row['연령대_40_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '50대': row['연령대_50_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '60대+': row['연령대_60_이상_유동인구_수']/total_flow*100 if total_flow>0 else 0
    }
    time_distribution = {
        '00-06시': row['시간대_00_06_유동인구_수'],
        '06-11시': row['시간대_06_11_유동인구_수'],
        '11-14시': row['시간대_11_14_유동인구_수'],
        '14-17시': row['시간대_14_17_유동인구_수'],
        '17-21시': row['시간대_17_21_유동인구_수'],
        '21-24시': row['시간대_21_24_유동인구_수']
    }
    population_ratio = {'유동인구': row['총_유동인구_수'], '상주인구': row['총_상주인구_수'], '직장인구': row['총_직장인구_수']}
    return {'age_distribution': age_distribution, 'time_distribution': time_distribution, 'population_ratio': population_ratio}

def get_income_consumption_stats(df, district, row):
    district_q2_2025 = df[(df['자치구_코드_명'] == district) & (df['기준_년분기_코드'] == 20252)]
    if len(district_q2_2025) > 0:
        q2_row = district_q2_2025.iloc[0]
        total_spending = q2_row['지출_총_금액'] / 1000
    else:
        total_spending = row['지출_총_금액'] / 1000
        q2_row = row
    spending_breakdown = {
        '식료품': (q2_row['식료품_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '음식': (q2_row['음식_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '의류/신발': (q2_row['의류_신발_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '생활용품': (q2_row['생활용품_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '의료비': (q2_row['의료비_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '교통': (q2_row['교통_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '교육': (q2_row['교육_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '유흥': (q2_row['유흥_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '여가/문화': (q2_row['여가_문화_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
    }
    return {'avg_income': row['월_평균_소득_금액'], 'total_spending': total_spending, 'spending_breakdown': spending_breakdown}

# === [ADD] 요일/시간 프로파일 계산 유틸 ===
DAY_KR = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
TIME_KEYS = [
    ("00-06시", "시간대_00_06_유동인구_수"),
    ("06-11시", "시간대_06_11_유동인구_수"),
    ("11-14시", "시간대_11_14_유동인구_수"),
    ("14-17시", "시간대_14_17_유동인구_수"),
    ("17-21시", "시간대_17_21_유동인구_수"),
    ("21-24시", "시간대_21_24_유동인구_수"),
]

def build_avg_day_profile_from_row(row: pd.Series) -> dict:
    """row에서 요일별 매출 금액을 퍼센트(0~100)로 정규화"""
    vals = [float(row.get(f"{d}_매출_금액", 0)) for d in DAY_KR]
    s = sum(vals)
    if s <= 0:
        return {d: 0.0 for d in DAY_KR}
    return {d: (v / s * 100.0) for d, v in zip(DAY_KR, vals)}

def build_avg_time_profile_from_row(row: pd.Series) -> dict:
    """row에서 시간대별 유동인구 값을 퍼센트(0~100)로 정규화"""
    vals = [float(row.get(col, 0)) for _, col in TIME_KEYS]
    s = sum(vals)
    labels = [lab for lab, _ in TIME_KEYS]
    if s <= 0:
        return {lab: 0.0 for lab in labels}
    return {lab: (v / s * 100.0) for lab, v in zip(labels, vals)}

def build_user_day_profile(selected_days: list[str]) -> dict:
    """사용자 선택 요일을 균등 가중으로 100% 분배 (미선택이면 전부 0)"""
    n = len(selected_days)
    base = {d: 0.0 for d in DAY_KR}
    if n == 0:
        return base
    share = 100.0 / n
    for d in selected_days:
        if d in base:
            base[d] = share
    return base

def build_user_time_profile(selected_times: list[str]) -> dict:
    """
    사용자 선택 시간대를 균등 가중으로 100% 분배 (미선택이면 전부 0)
    selected_times는 ['00_06','06_11',...] 형태이므로 라벨로 변환해 비교
    """
    label_map = {
        "00_06": "00-06시", "06_11": "06-11시", "11_14": "11-14시",
        "14_17": "14-17시", "17_21": "17-21시", "21_24": "21-24시"
    }
    labels = [label_map.get(t, t) for t in selected_times]
    all_labels = [lab for lab, _ in TIME_KEYS]
    base = {lab: 0.0 for lab in all_labels}
    n = len(labels)
    if n == 0:
        return base
    share = 100.0 / n
    for lab in labels:
        if lab in base:
            base[lab] = share
    return base

# ==================== 입력 폼 ====================
st.markdown("### 매장 정보를 입력해주세요")
st.markdown("---")

merged_df = load_merged_data()
if merged_df is None:
    st.error("데이터를 불러올 수 없습니다. 관리자에게 문의하세요.")
    st.stop()

districts = sorted(merged_df['자치구_코드_명'].unique())
industries = sorted(merged_df['서비스_업종_코드_명'].unique())

col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='info-box'><h4>기본 정보</h4></div>", unsafe_allow_html=True)
    selected_district = st.selectbox("자치구 선택", options=districts)
    selected_industry = st.selectbox("서비스 업종 선택", options=industries)

with col2:
    st.markdown("<div class='info-box'><h4>재무 정보</h4></div>", unsafe_allow_html=True)
    # TODO: 현재는 결과 미반영 (입력만 저장)
    input_sales = st.number_input("당월 매출 금액 (원)", min_value=0, max_value=1_000_000_000, value=0, step=10_000,
                                  help="※ 지금은 결과에 반영되지 않습니다(입력만 저장). 추후 반영 로직 설계 예정.")
    rent = st.number_input("월 임대료 (원)", min_value=0, max_value=100_000_000, value=3_000_000, step=100_000,
                           help="임대료는 임대료 부담률/예측에 반영됩니다.")

st.markdown("<br>", unsafe_allow_html=True)

# 영업 요일/시간 (입력만, 결과 미반영)
st.markdown("### 영업 요일/시간 (입력 전용)")
cl, cr = st.columns(2)
DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TIME_BUCKETS = ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]
with cl:
    st.markdown("<div class='info-box'><h4>영업 요일 (다중 선택)</h4></div>", unsafe_allow_html=True)
    selected_days = st.multiselect("영업하는 요일을 선택하세요", DAYS,
                                   help="※ 지금은 결과에 반영되지 않습니다(입력만 저장). 추후 반영 로직 설계 예정.")
with cr:
    st.markdown("<div class='info-box'><h4>영업 시간대 (다중 선택)</h4></div>", unsafe_allow_html=True)
    selected_times = st.multiselect("영업하는 시간대를 선택하세요", TIME_BUCKETS,
                                    help="※ 지금은 결과에 반영되지 않습니다(입력만 저장). 추후 반영 로직 설계 예정.")

st.markdown("---")

# ==================== Session State ====================
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# ==================== 예측 버튼 ====================
if st.button("폐업 위험도 예측하기", type="primary"):
    with st.spinner("AI가 데이터를 분석하고 있습니다..."):
        model, district_encoder, industry_encoder, sanggwon_encoder, feature_names = load_model_and_encoders()
        if model is None:
            st.error("모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
            st.stop()

        row_data = get_district_industry_data(merged_df, selected_district, selected_industry)
        if row_data is None:
            st.error(f"{selected_district} - {selected_industry} 데이터가 없습니다. 다른 조합을 선택해주세요.")
            st.stop()

        stats = calculate_statistics(row_data)

        # 예측용 입력: 임대료만 반영 (매출/요일/시간 미반영 유지)
        input_data = row_data.copy()
        if '전체임대료' in input_data:
            input_data['전체임대료'] = rent
        elif '임대료' in input_data:
            input_data['임대료'] = rent

        # 인코딩
        try:
            input_data['자치구_코드_명'] = district_encoder.transform([selected_district])[0]
            input_data['서비스_업종_코드_명'] = industry_encoder.transform([selected_industry])[0]
            if '상권_변화_지표' not in input_data or pd.isna(input_data['상권_변화_지표']):
                input_data['상권_변화_지표'] = "보합"
            input_data['상권_변화_지표'] = sanggwon_encoder.transform([input_data['상권_변화_지표']])[0]
        except Exception as e:
            st.error(f"인코딩 오류: {str(e)}")
            st.stop()

        # 불필요 컬럼 제거(학습 제외 컬럼)
        drop_cols = ['기준_년분기_코드', '폐업_점포_수', '폐업_영업_개월_평균', '서울시_폐업_영업_개월_평균', '폐업_률']
        input_df = pd.DataFrame([input_data])
        X_input = input_df.drop(columns=[c for c in drop_cols if c in input_df.columns], errors='ignore')

        # 예측
        try:
            prediction_proba = model.predict_proba(X_input)[:, 1][0]
            risk_score = float(prediction_proba) * 100
        except Exception as e:
            st.error(f"예측 오류: {str(e)}")
            st.error(f"입력 피처 개수: {len(X_input.columns)}")
            st.stop()

        # SessionState 저장 (+ 이식된 비교/통계 전부)
        st.session_state.prediction_done = True
        st.session_state.selected_district = selected_district
        st.session_state.selected_industry = selected_industry
        st.session_state.rent = rent
        st.session_state.input_sales = input_sales   # TODO: 아직 미반영
        st.session_state.selected_days = selected_days  # TODO: 아직 미반영
        st.session_state.selected_times = selected_times # TODO: 아직 미반영
        st.session_state.row_data = row_data
        st.session_state.stats = stats
        st.session_state.risk_score = risk_score
        st.session_state.industry_comp = get_industry_comparison(merged_df, selected_industry, selected_district)
        st.session_state.district_comp = get_district_comparison(merged_df, selected_district)

# ==================== 결과 표시 (예비 창업자 섹션 전부 이식) ====================
if st.session_state.prediction_done:
    selected_district = st.session_state.selected_district
    selected_industry = st.session_state.selected_industry
    rent = st.session_state.rent
    input_sales = st.session_state.input_sales
    selected_days = st.session_state.selected_days
    selected_times = st.session_state.selected_times
    row_data = st.session_state.row_data
    stats = st.session_state.stats
    risk_score = st.session_state.risk_score
    industry_comp = st.session_state.industry_comp
    district_comp = st.session_state.district_comp

    st.markdown("### 분석 결과")
    st.markdown("---")

    # 위험도 레벨
    if risk_score >= 70:
        risk_level, risk_color, risk_emoji, message_class = "높음", "#dc3545", "🚨", "danger-box"
        message = "현재 입력하신 조건은 폐업 위험이 높은 편입니다. 신중한 검토가 필요합니다."
    elif risk_score >= 40:
        risk_level, risk_color, risk_emoji, message_class = "보통", "#ffc107", "⚠️", "warning-box"
        message = "현재 입력하신 조건은 보통 수준의 위험도를 보이고 있습니다."
    else:
        risk_level, risk_color, risk_emoji, message_class = "낮음", "#28a745", "✅", "success-box"
        message = "현재 입력하신 조건은 비교적 안정적인 편입니다."

    # 게이지
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{risk_emoji} 폐업 위험도", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font={'color': "darkblue", 'family': "Arial"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="{message_class}">
        <h3>{risk_emoji} 위험도: {risk_level} ({risk_score:.1f}점)</h3>
        <p style='margin:0; font-size: 1.05rem;'>{message}</p>
    </div>
    """, unsafe_allow_html=True)

    # 상세 카드
    expected_sales = stats['평균_매출']
    rent_burden = (rent / expected_sales * 100) if expected_sales > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                임대료 부담률 <span class="tooltip-icon" data-tooltip="매출 대비 임대료 비율 (적정: 10% 이하)">ℹ️</span>
            </div>
            <div class="stat-value">{rent_burden:.1f}%</div>
            <div style="color: {'#dc3545' if rent_burden > 15 else '#ffc107' if rent_burden > 10 else '#28a745'};">
                {'🚨 높음' if rent_burden > 15 else '⚠️ 주의' if rent_burden > 10 else '✅ 적정'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">지역 평균 매출</div>
            <div class="stat-value">{stats['평균_매출']/100000000:,.1f}억</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">지역 평균 건수</div>
            <div class="stat-value">{stats['평균_매출건수']:,.0f}건</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">폐업률</div>
            <div class="stat-value">{row_data['폐업_률']:.1f}%</div>
            <div style="color: #6c757d;">해당 지역</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 지역 통계
    st.markdown(f"### {selected_district} - {selected_industry} 통계")
    col_left, col_right = st.columns(2)
    with col_left:
        fig_sales = go.Figure(data=[
            go.Bar(x=['주말 매출', '주중 매출'],
                   y=[stats['주말_매출_비율'], 100 - stats['주말_매출_비율']],
                   marker_color=['#667eea', '#764ba2'],
                   text=[f"{stats['주말_매출_비율']:.1f}%", f"{100-stats['주말_매출_비율']:.1f}%"],
                   textposition='auto')
        ])
        fig_sales.update_layout(title="주중/주말 매출 비율", yaxis_title="비율 (%)", height=300, showlegend=False)
        st.plotly_chart(fig_sales, use_container_width=True)
    with col_right:
        fig_gender = go.Figure(data=[
            go.Pie(labels=['남성', '여성'],
                   values=[stats['남성_매출_비율'], stats['여성_매출_비율']],
                   marker_colors=['#667eea', '#764ba2'],
                   textinfo='label+percent')
        ])
        fig_gender.update_layout(title="성별 매출 비율", height=300)
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    # 업종 비교 분석
    st.markdown(f"### 업종 비교 분석: {selected_industry}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">서울시 평균 매출</div>
            <div class="stat-value">{industry_comp['seoul_avg_sales']/100000000:,.1f}억</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        sales_vs_seoul = (industry_comp['district_sales'] / industry_comp['seoul_avg_sales'] * 100) if industry_comp['seoul_avg_sales'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                서울 평균 대비 <span class="tooltip-icon" data-tooltip="선택 지역 매출이 서울시 평균의 몇 %인지 표시">ℹ️</span>
            </div>
            <div class="stat-value">{sales_vs_seoul:,.0f}%</div>
            <div style="color: {'#28a745' if sales_vs_seoul >= 100 else '#dc3545'};">
                {'평균 이상' if sales_vs_seoul >= 100 else '평균 이하'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">매출 순위 <span class="tooltip-icon" data-tooltip="해당 업종에서 25개 자치구 중 매출 순위">ℹ️</span></div>
            <div class="stat-value">{industry_comp['sales_rank']}</div>
            <div style="color: #6c757d;">/ {industry_comp['total_districts']}개 자치구</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">안전도 순위 <span class="tooltip-icon" data-tooltip="폐업률이 낮을수록 순위가 높음 (1위=가장 안전)">ℹ️</span></div>
            <div class="stat-value">{industry_comp['closure_rank']}</div>
            <div style="color: #6c757d;">낮은 폐업률 기준</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 자치구 업종 분석
    st.markdown(f"### 자치구 업종 분석: {selected_district}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 매출 상위 5개 업종")
        for _, r in district_comp['top_sales'].iterrows():
            st.markdown(f"**{r['서비스_업종_코드_명']}**")
            st.markdown(f"매출: {r['당월_매출_금액']/100000000:,.1f}억원 | 폐업률: {r['폐업_률']:.1f}%")
            st.markdown("---")
    with col2:
        st.markdown("#### 안전한 업종 5개")
        st.caption("폐업률 기준")
        for _, r in district_comp['safe_industries'].iterrows():
            st.markdown(f"**{r['서비스_업종_코드_명']}**")
            st.markdown(f"폐업률: {r['폐업_률']:.1f}% | 매출: {r['당월_매출_금액']/100000000:,.1f}억원")
            st.markdown("---")
    with col3:
        st.markdown("#### 위험한 업종 5개")
        st.caption("폐업률 기준")
        for _, r in district_comp['risky_industries'].iterrows():
            st.markdown(f"**{r['서비스_업종_코드_명']}**")
            st.markdown(f"폐업률: {r['폐업_률']:.1f}% | 매출: {r['당월_매출_금액']/100000000:,.1f}억원")
            st.markdown("---")

    st.markdown("---")

    # 분기별 트렌드
    st.markdown("### 분기별 트렌드")
    available_years = get_available_years(merged_df)
    col_year, _ = st.columns([1, 3])
    with col_year:
        selected_year = st.selectbox("연도 선택", options=available_years, index=len(available_years)-1)
    time_series = get_time_series_data(merged_df, selected_district, selected_industry, selected_year)
    if len(time_series['quarters']) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig_sales_trend = go.Figure()
            fig_sales_trend.add_trace(go.Scatter(x=time_series['quarters'], y=time_series['sales'],
                                                 mode='lines+markers', name='매출',
                                                 line=dict(color='#667eea', width=3), marker=dict(size=10)))
            fig_sales_trend.update_layout(title=f"{selected_year}년 분기별 매출 추이", xaxis_title="분기", yaxis_title="매출 (억원)", height=300)
            st.plotly_chart(fig_sales_trend, use_container_width=True)
        with col2:
            fig_closure_trend = go.Figure()
            fig_closure_trend.add_trace(go.Scatter(x=time_series['quarters'], y=time_series['closure_rate'],
                                                   mode='lines+markers', name='폐업률',
                                                   line=dict(color='#dc3545', width=3), marker=dict(size=10)))
            fig_closure_trend.update_layout(title=f"{selected_year}년 분기별 폐업률 변화", xaxis_title="분기", yaxis_title="폐업률 (%)", height=300)
            st.plotly_chart(fig_closure_trend, use_container_width=True)
        fig_store_trend = go.Figure()
        fig_store_trend.add_trace(go.Bar(x=time_series['quarters'], y=time_series['store_count'],
                                         marker_color='#764ba2',
                                         text=[f"{int(v):,}" for v in time_series['store_count']], textposition='auto'))
        fig_store_trend.update_layout(title=f"{selected_year}년 분기별 점포 수 변화", xaxis_title="분기", yaxis_title="점포 수", height=300)
        st.plotly_chart(fig_store_trend, use_container_width=True)
    else:
        st.warning("시계열 데이터가 부족합니다.")

    st.markdown("---")

    # 인구 통계
    st.markdown("### 인구 통계")
    pop_stats = get_population_stats(row_data)
    col1, col2 = st.columns(2)
    with col1:
        fig_age = go.Figure(data=[go.Bar(
            x=list(pop_stats['age_distribution'].keys()),
            y=list(pop_stats['age_distribution'].values()),
            marker_color='#667eea',
            text=[f"{v:.1f}%" for v in pop_stats['age_distribution'].values()],
            textposition='auto'
        )])
        fig_age.update_layout(title="연령대별 유동인구 비율", xaxis_title="연령대", yaxis_title="비율 (%)", height=300)
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        fig_time = go.Figure(data=[go.Bar(
            x=list(pop_stats['time_distribution'].keys()),
            y=list(pop_stats['time_distribution'].values()),
            marker_color='#764ba2',
            text=[f"{v:,.0f}" for v in pop_stats['time_distribution'].values()],
            textposition='auto'
        )])
        fig_time.update_layout(title="시간대별 유동인구", xaxis_title="시간대", yaxis_title="유동인구 수", height=300)
        st.plotly_chart(fig_time, use_container_width=True)

    fig_pop_ratio = go.Figure(data=[go.Pie(
        labels=list(pop_stats['population_ratio'].keys()),
        values=list(pop_stats['population_ratio'].values()),
        textinfo='label+percent'
    )])
    fig_pop_ratio.update_layout(title="인구 구성 (유동/상주/직장)", height=350)
    st.plotly_chart(fig_pop_ratio, use_container_width=True)

    st.markdown("---")

    # =================== 소득/지출 분석 (내 당월 매출 카드 추가) ===================
    st.markdown("### 소득/지출 분석")
    income_stats = get_income_consumption_stats(merged_df, selected_district, row_data)

    # 왼쪽 칼럼에 카드 3개: (1) 내 당월 매출 (2) 개인 월 평균 소득 (3) 지역 총 지출
    col_left, col_right = st.columns([1, 2.3])

    with col_left:
        # (1) 내 당월 매출 금액
        st.markdown(f"""
        <div class="stat-card compact">
            <div class="stat-label">내 당월 매출 금액</div>
            <div class="stat-value">{input_sales:,.0f}원</div>
            <div style="color: #6c757d;">입력값 기준</div>
        </div>
        """, unsafe_allow_html=True)

        # (2) 개인 월 평균 소득
        st.markdown(f"""
        <div class="stat-card compact">
            <div class="stat-label">개인 월 평균 소득</div>
            <div class="stat-value">{income_stats['avg_income']/10000:,.0f}만</div>
            <div style="color: #6c757d;">개인 월 기준</div>
        </div>
        """, unsafe_allow_html=True)

        # (3) 지역 총 지출
        st.markdown(f"""
        <div class="stat-card compact">
            <div class="stat-label">지역 총 지출 <span class="tooltip-icon" data-tooltip="2025년 2분기 기준 지역 전체 총 지출 금액">ℹ️</span></div>
            <div class="stat-value">{income_stats['total_spending']/100000000:,.0f}억</div>
            <div style="color: #6c757d;">2025년 2분기 기준</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # 지출 항목별 비율 (그대로 유지)
        fig_spending = go.Figure(data=[go.Bar(
            y=list(income_stats['spending_breakdown'].keys()),
            x=list(income_stats['spending_breakdown'].values()),
            orientation='h',
            marker_color='#667eea',
            text=[f"{v:.1f}%" for v in income_stats['spending_breakdown'].values()],
            textposition='auto'
        )])
        fig_spending.update_layout(title="지출 항목별 비율", xaxis_title="비율 (%)", yaxis_title="항목", height=350)
        st.plotly_chart(fig_spending, use_container_width=True)

    st.markdown("---")

    # =================== 영업 요일/시간 비교 ===================
    st.markdown("---")
    st.markdown("### 영업 요일/시간 비교")

    # 지역 평균(=선택한 자치구+업종의 대표 행) 프로파일
    avg_day_profile  = build_avg_day_profile_from_row(row_data)      # 요일: 매출 비중 기반
    avg_time_profile = build_avg_time_profile_from_row(row_data)     # 시간: 유동인구 비중 기반

    # 내 선택 프로파일 (균등가중 100% 분배)
    user_day_profile  = build_user_day_profile(selected_days)
    user_time_profile = build_user_time_profile(selected_times)

    # =================== 영업 요일/시간 비교 ===================
    st.markdown("---")
    st.markdown("### 영업 요일/시간 비교")

    # 지역 평균(=선택한 자치구+업종의 대표 행) 프로파일
    avg_day_profile  = build_avg_day_profile_from_row(row_data)      # 요일: 매출 비중 기반(%)
    avg_time_profile = build_avg_time_profile_from_row(row_data)     # 시간: 유동 비중 기반(%)

    # 선택 항목 집합
    days = DAY_KR  # ["월요일",..., "일요일"]
    selected_day_set = set(selected_days or [])

    label_map = {
        "00_06": "00-06시", "06_11": "06-11시", "11_14": "11-14시",
        "14_17": "14-17시", "17_21": "17-21시", "21_24": "21-24시"
    }
    time_labels = [lab for lab, _ in TIME_KEYS]
    selected_time_set = set(label_map.get(t, t) for t in (selected_times or []))

    # 색상(선택=진한색, 미선택=연한색)
    COLOR_SELECTED = "#3b82f6"   # 진한 파랑
    COLOR_DEFAULT  = "#93c5fd"   # 연한 파랑

    # ── 요일 비교: 하나의 trace에 막대 색만 다르게 ───────────────────────────────
    y_day = [avg_day_profile[d] for d in days]
    colors_day = [COLOR_SELECTED if d in selected_day_set else COLOR_DEFAULT for d in days]

    fig_days = go.Figure()
    fig_days.add_trace(go.Bar(
        x=days, y=y_day,
        marker_color=colors_day,
        text=[f"{v:.1f}%" for v in y_day],
        textposition="auto",
        hovertemplate="요일: %{x}<br>지역 평균: %{y:.1f}%<extra></extra>",
        name="지역 평균(매출 비중)",
        showlegend=False
    ))
    # 범례용 더미(선택/기본 색 설명)
    fig_days.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_SELECTED, name="내가 선택한 요일"))
    fig_days.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_DEFAULT,  name="선택하지 않은 요일"))

    fig_days.update_layout(
        title="요일별 비교",
        yaxis_title="비율 (%)",
        barmode="overlay",
        height=360,
        showlegend=True
    )
    st.plotly_chart(fig_days, use_container_width=True)

    # ── 시간대 비교: 하나의 trace에 막대 색만 다르게 ────────────────────────────
    y_time = [avg_time_profile[lab] for lab in time_labels]
    colors_time = [COLOR_SELECTED if lab in selected_time_set else COLOR_DEFAULT for lab in time_labels]

    fig_timecmp = go.Figure()
    fig_timecmp.add_trace(go.Bar(
        x=time_labels, y=y_time,
        marker_color=colors_time,
        text=[f"{v:.1f}%" for v in y_time],
        textposition="auto",
        hovertemplate="시간대: %{x}<br>지역 평균: %{y:.1f}%<extra></extra>",
        name="지역 평균(유동 비중)",
        showlegend=False
    ))
    # 범례용 더미
    fig_timecmp.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_SELECTED, name="내가 선택한 시간대"))
    fig_timecmp.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_DEFAULT,  name="선택하지 않은 시간대"))

    fig_timecmp.update_layout(
        title="시간대별 비교",
        yaxis_title="비율 (%)",
        barmode="overlay",
        height=360,
        showlegend=True
    )
    st.plotly_chart(fig_timecmp, use_container_width=True)

    # 보조 설명
    st.caption("• 선택한 항목은 막대 색상으로만 **강조**하고, 수치는 지역 평균 비중을 표시합니다.")
    st.caption("• 선택 항목이 없으면 모든 막대가 기본색으로 표시됩니다.")

    # 권장 액션 리스트 (예비 창업자 섹션과 동일)
    st.markdown("### 권장 액션")
    recommendations = []
    if rent_burden > 15:
        recommendations.append("🔴 **임대료 부담률이 높습니다.** 임대료 재협상/대체 입지를 검토하세요.")
    elif rent_burden > 10:
        recommendations.append("🟡 **임대료 부담률이 다소 높습니다.** 매출 개선/비용 절감을 병행하세요.")
    else:
        recommendations.append("🟢 **임대료 부담률이 적정 수준입니다.**")

    if risk_score >= 70:
        recommendations.append("🔴 **폐업 위험도가 높습니다.** 전략 재검토 또는 대안 고려가 필요합니다.")
    elif risk_score >= 40:
        recommendations.append("🟡 **폐업 위험도가 보통입니다.** 차별화 전략을 고민하세요.")

    if stats['주말_매출_비율'] > 40:
        recommendations.append("📊 주말 매출 비중이 높습니다. 주말 운영 전략을 강화하세요.")
    if row_data['폐업_률'] > 5:
        recommendations.append("⚠️ 지역 폐업률이 높은 편입니다. 경쟁 환경 분석을 강화하세요.")
    recommendations.append("📍 상권 특성/유동인구를 추가로 분석해보세요.")
    recommendations.append("🤝 전문가 상담을 통해 더 정확한 경영계획을 수립하세요.")

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# 하단 안내/푸터
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>ℹ️ 안내사항</h4>
    <ul style='line-height: 2;'>
        <li>본 서비스는 AI 기반 예측 결과로, 참고용으로만 활용하시기 바랍니다.</li>
        <li>실제 운영/확장 결정 시에는 전문가 상담과 충분한 시장조사가 필요합니다.</li>
        <li>예측 결과는 과거 데이터를 기반으로 계산되며, 실제 결과와 다를 수 있습니다.</li>
        <li>시장 상황, 경쟁 환경 등 다양한 외부 요인도 함께 고려하셔야 합니다.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;'>
    <p>© 2025 서울시 자치구별 매장 폐업 예측 프로젝트. All rights reserved.</p>
    <p>매장 운영자를 위한 AI 기반 의사결정 지원 서비스</p>
</div>
""", unsafe_allow_html=True)
