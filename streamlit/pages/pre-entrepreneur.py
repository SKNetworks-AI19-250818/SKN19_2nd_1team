import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="예비 창업자를 위한 폐업 예측",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }

    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }

    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; font-size: 2.5rem;'>예비 창업자를 위한 폐업 위험 예측</h1>
    <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
        AI 기반 데이터 분석으로 창업 전 폐업 위험도를 미리 확인하세요
    </p>
</div>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("메뉴")
st.sidebar.markdown("---")
st.sidebar.info("""
**예비 창업자 지원 서비스**

창업을 계획 중이신가요?
AI가 예상 폐업 위험도를
분석해드립니다!
""")

# ==================== 데이터 로드 함수 ====================

@st.cache_data
def load_merged_data():
    """merged_data.csv 로드"""
    try:
        df = pd.read_csv('../model/catboost/data/merged_data.csv')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

@st.cache_resource
def load_model_and_encoders():
    """모델 및 인코더 로드"""
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
    """특정 자치구+업종의 최신 데이터 가져오기"""
    filtered = df[
        (df['자치구_코드_명'] == district) &
        (df['서비스_업종_코드_명'] == industry)
    ]

    if len(filtered) == 0:
        return None

    # 최신 데이터 반환
    latest = filtered.sort_values('기준_년분기_코드', ascending=False).iloc[0]
    return latest

def calculate_statistics(row):
    """통계 계산"""
    if row is None:
        return None

    stats = {}

    # 매출 관련
    if row['당월_매출_금액'] > 0:
        stats['평균_매출'] = row['당월_매출_금액']
        stats['평균_매출건수'] = row['당월_매출_건수']
        stats['주말_매출_비율'] = ((row['토요일_매출_금액'] + row['일요일_매출_금액']) / row['당월_매출_금액'] * 100)
        stats['남성_매출_비율'] = (row['남성_매출_금액'] / row['당월_매출_금액'] * 100)
        stats['여성_매출_비율'] = (row['여성_매출_금액'] / row['당월_매출_금액'] * 100)
    else:
        stats['평균_매출'] = 0
        stats['평균_매출건수'] = 0
        stats['주말_매출_비율'] = 0
        stats['남성_매출_비율'] = 0
        stats['여성_매출_비율'] = 0

    return stats

# ==================== 메인 화면 ====================

st.markdown("### 창업 정보를 입력해주세요")
st.markdown("---")

# 데이터 로드
merged_df = load_merged_data()

if merged_df is None:
    st.error("데이터를 불러올 수 없습니다. 관리자에게 문의하세요.")
    st.stop()

# 자치구/업종 리스트 추출
districts = sorted(merged_df['자치구_코드_명'].unique())
industries = sorted(merged_df['서비스_업종_코드_명'].unique())

# 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h4>기본 정보</h4>
    </div>
    """, unsafe_allow_html=True)

    selected_district = st.selectbox(
        "자치구 선택",
        options=districts,
        help="창업을 계획하고 계신 자치구를 선택하세요"
    )

    selected_industry = st.selectbox(
        "서비스 업종 선택",
        options=industries,
        help="창업을 계획하고 계신 업종을 선택하세요"
    )

with col2:
    st.markdown("""
    <div class="info-box">
        <h4>재무 정보</h4>
    </div>
    """, unsafe_allow_html=True)

    rent = st.number_input(
        "월 임대료 (원)",
        min_value=0,
        max_value=100000000,
        value=3000000,
        step=100000,
        help="예상하시는 월 임대료를 입력하세요"
    )

st.markdown("---")

# 예측 버튼
if st.button("폐업 위험도 예측하기", type="primary"):
    with st.spinner("AI가 데이터를 분석하고 있습니다..."):

        # 모델 및 인코더 로드
        model, district_encoder, industry_encoder, sanggwon_encoder, feature_names = load_model_and_encoders()

        if model is None:
            st.error("모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
            st.stop()

        # 해당 자치구+업종 데이터 가져오기
        row_data = get_district_industry_data(merged_df, selected_district, selected_industry)

        if row_data is None:
            st.error(f"{selected_district} - {selected_industry}에 대한 데이터가 없습니다. 다른 조합을 선택해주세요.")
            st.stop()

        # 통계 계산
        stats = calculate_statistics(row_data)

        # 예측용 데이터 준비
        input_data = row_data.copy()

        # 사용자 입력 임대료로 교체
        input_data['전체임대료'] = rent

        # 인코딩
        try:
            input_data['자치구_코드_명'] = district_encoder.transform([selected_district])[0]
            input_data['서비스_업종_코드_명'] = industry_encoder.transform([selected_industry])[0]
            input_data['상권_변화_지표'] = sanggwon_encoder.transform([input_data['상권_변화_지표']])[0]
        except Exception as e:
            st.error(f"인코딩 오류: {str(e)}")
            st.stop()

        # 불필요한 컬럼 제거 (학습 때와 동일)
        drop_cols = ['기준_년분기_코드', '폐업_점포_수', '폐업_영업_개월_평균',
                     '서울시_폐업_영업_개월_평균', '폐업_률']

        # DataFrame으로 변환
        input_df = pd.DataFrame([input_data])

        # 제거할 컬럼만 제거 (존재하는 것만)
        cols_to_drop = [col for col in drop_cols if col in input_df.columns]
        X_input = input_df.drop(columns=cols_to_drop)

        # 예측
        try:
            prediction_proba = model.predict_proba(X_input)[:, 1][0]
            risk_score = prediction_proba * 100
        except Exception as e:
            st.error(f"예측 오류: {str(e)}")
            st.error(f"입력 피처 개수: {len(X_input.columns)}")
            st.error(f"모델 기대 피처: {feature_names[:5]}...")
            st.stop()

        # ==================== 결과 표시 ====================

        st.markdown("### 분석 결과")
        st.markdown("---")

        # 위험도 레벨 결정
        if risk_score >= 70:
            risk_level = "높음"
            risk_color = "#dc3545"
            risk_emoji = "🚨"
            message_class = "danger-box"
            message = "현재 입력하신 조건은 폐업 위험이 높은 편입니다. 신중한 검토가 필요합니다."
        elif risk_score >= 40:
            risk_level = "보통"
            risk_color = "#ffc107"
            risk_emoji = "⚠️"
            message_class = "warning-box"
            message = "현재 입력하신 조건은 보통 수준의 위험도를 보이고 있습니다."
        else:
            risk_level = "낮음"
            risk_color = "#28a745"
            risk_emoji = "✅"
            message_class = "success-box"
            message = "현재 입력하신 조건은 비교적 안정적인 편입니다."

        # 게이지 차트
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{risk_emoji} 폐업 위험도", 'font': {'size': 24}},
            delta={'reference': 50},
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
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "darkblue", 'family': "Arial"}
        )

        st.plotly_chart(fig, use_container_width=True)

        # 위험도 메시지
        st.markdown(f"""
        <div class="{message_class}">
            <h3>{risk_emoji} 위험도: {risk_level} ({risk_score:.1f}점)</h3>
            <p style='margin:0; font-size: 1.1rem;'>{message}</p>
        </div>
        """, unsafe_allow_html=True)

        # 상세 분석
        st.markdown("### 상세 분석")

        col1, col2, col3, col4 = st.columns(4)

        # 임대료 부담률
        expected_sales = stats['평균_매출']
        rent_burden = (rent / expected_sales * 100) if expected_sales > 0 else 0

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">임대료 부담률</div>
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
                <div class="stat-value">{stats['평균_매출']/10000:.0f}만</div>
                <div style="color: #6c757d;">월 기준</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">지역 평균 건수</div>
                <div class="stat-value">{stats['평균_매출건수']:.0f}건</div>
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
        st.markdown(f"### 📍 {selected_district} - {selected_industry} 통계")

        col_left, col_right = st.columns(2)

        with col_left:
            # 주중/주말 매출 비율
            fig_sales = go.Figure(data=[
                go.Bar(
                    x=['주말 매출', '주중 매출'],
                    y=[stats['주말_매출_비율'], 100 - stats['주말_매출_비율']],
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"{stats['주말_매출_비율']:.1f}%", f"{100-stats['주말_매출_비율']:.1f}%"],
                    textposition='auto',
                )
            ])
            fig_sales.update_layout(
                title="주중/주말 매출 비율",
                yaxis_title="비율 (%)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_sales, use_container_width=True)

        with col_right:
            # 성별 매출 비율
            fig_gender = go.Figure(data=[
                go.Pie(
                    labels=['남성', '여성'],
                    values=[stats['남성_매출_비율'], stats['여성_매출_비율']],
                    marker_colors=['#667eea', '#764ba2'],
                    textinfo='label+percent',
                )
            ])
            fig_gender.update_layout(
                title="성별 매출 비율",
                height=300
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # 권장사항
        st.markdown("### 💡 권장사항")

        recommendations = []

        if rent_burden > 15:
            recommendations.append("🔴 **임대료 부담률이 매우 높습니다.** 더 저렴한 임대료의 매장을 찾아보시거나, 매출 증대 방안을 고려하세요.")
        elif rent_burden > 10:
            recommendations.append("🟡 **임대료 부담률이 다소 높은 편입니다.** 매출 증대 또는 비용 절감 방안을 준비하세요.")
        else:
            recommendations.append("🟢 **임대료 부담률이 적정 수준입니다.**")

        if risk_score >= 70:
            recommendations.append("🔴 **폐업 위험도가 높습니다.** 창업 계획을 재검토하거나 다른 지역/업종을 고려해보세요.")
        elif risk_score >= 40:
            recommendations.append("🟡 **폐업 위험도가 보통입니다.** 차별화된 전략이 필요합니다.")

        if stats['주말_매출_비율'] > 40:
            recommendations.append("📊 이 업종은 주말 매출 비중이 높습니다. 주말 영업 전략을 중점적으로 준비하세요.")

        if row_data['폐업_률'] > 5:
            recommendations.append("⚠️ 해당 지역의 폐업률이 높은 편입니다. 경쟁 환경을 신중히 분석하세요.")

        recommendations.append("📍 해당 지역의 상권 특성과 유동인구를 추가로 분석해보시기 바랍니다.")
        recommendations.append("🤝 전문가 상담을 통해 더 정확한 사업계획을 수립하시는 것을 권장합니다.")

        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

# 하단 정보
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>ℹ️ 안내사항</h4>
    <ul style='line-height: 2;'>
        <li>본 서비스는 AI 기반 예측 결과로, 참고용으로만 활용하시기 바랍니다.</li>
        <li>실제 창업 결정 시에는 전문가 상담과 충분한 시장조사가 필요합니다.</li>
        <li>예측 결과는 과거 데이터를 기반으로 계산되며, 실제 결과와 다를 수 있습니다.</li>
        <li>시장 상황, 경쟁 환경 등 다양한 외부 요인도 함께 고려하셔야 합니다.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 푸터
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;'>
    <p>© 2025 서울시 자치구별 매장 폐업 예측 프로젝트. All rights reserved.</p>
    <p>예비 창업자를 위한 AI 기반 의사결정 지원 서비스</p>
</div>
""", unsafe_allow_html=True)
