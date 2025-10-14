import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="서울시 자치구별 매장 폐업 예측",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 헤더 스타일 */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* 통계 카드 */
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
    
    .stat-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .positive {
        color: #28a745;
    }
    
    .negative {
        color: #dc3545;
    }
    
    /* 정보 박스 */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* 버튼 스타일 */
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

# 사이드바 네비게이션
st.sidebar.title("📊 메뉴")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "페이지 선택",
    ["🏠 홈", "📈 페이지 2 (준비중)", "🗺️ 페이지 3 (준비중)", "📋 페이지 4 (준비중)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**서울시 자치구별 매장 폐업 예측**

AI 기반 데이터 분석으로
서울시 각 자치구의 매장 폐업 위험을
예측하고 분석합니다.
""")

# 메인 헤더
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; font-size: 2.5rem;'>🏪 서울시 자치구별 매장 폐업 예측</h1>
    <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
        AI 기반 데이터 분석으로 예측하는 서울시 자치구별 상권 현황
    </p>
</div>
""", unsafe_allow_html=True)

# 현재 날짜 표시
current_date = datetime.now()
st.markdown(f"### 📅 {current_date.year}년 {current_date.month}월 기준")
st.markdown("---")

# 주요 통계 (샘플 데이터)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">전체 분석 매장 수</div>
        <div class="stat-value">125,847</div>
        <div class="stat-change positive">▲ 전분기 대비 +2,340</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">폐업 위험 매장</div>
        <div class="stat-value">8,234</div>
        <div class="stat-change negative">▲ 전분기 대비 +523</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">평균 폐업 위험도</div>
        <div class="stat-value">6.5%</div>
        <div class="stat-change positive">▼ 전분기 대비 -0.3%p</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">분석 자치구 수</div>
        <div class="stat-value">25</div>
        <div class="stat-change">서울시 전체</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 메인 컨텐츠
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 프로젝트 소개</h3>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #495057;'>
            본 프로젝트는 서울시 25개 자치구의 상권 데이터를 기반으로
            <strong>머신러닝 알고리즘</strong>을 활용하여 매장의 폐업 위험도를 예측합니다.
        </p>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #495057;'>
            다양한 지표(유동인구, 매출액, 임대료, 업종 등)를 종합적으로 분석하여
            사업자들에게 유용한 인사이트를 제공하는 것을 목표로 합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 자치구별 폐업 위험도 차트 (샘플)
    st.markdown("""
    <div class="info-box">
        <h3>📊 자치구별 폐업 위험도 Top 10</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 샘플 데이터
    districts = ['중구', '종로구', '용산구', '성동구', '광진구', 
                 '동대문구', '중랑구', '성북구', '강북구', '도봉구']
    risk_rates = [8.5, 8.2, 7.9, 7.6, 7.3, 7.1, 6.9, 6.7, 6.5, 6.3]
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_rates,
            y=districts,
            orientation='h',
            marker=dict(
                color=risk_rates,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="위험도 (%)")
            ),
            text=[f'{rate}%' for rate in risk_rates],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        xaxis_title="폐업 위험도 (%)",
        yaxis_title="자치구",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("""
    <div class="info-box">
        <h3>🔍 주요 기능</h3>
        <ul style='font-size: 1rem; line-height: 2; color: #495057;'>
            <li>자치구별 폐업 위험도 예측</li>
            <li>업종별 생존율 분석</li>
            <li>상권 트렌드 시각화</li>
            <li>맞춤형 리포트 생성</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📈 분석 지표</h3>
        <ul style='font-size: 1rem; line-height: 2; color: #495057;'>
            <li>유동인구 데이터</li>
            <li>매출액 정보</li>
            <li>임대료 수준</li>
            <li>업종 경쟁도</li>
            <li>지역 경제 지표</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <h3 style="color: white;">💡 사용 가이드</h3>
        <p style='font-size: 0.95rem; line-height: 1.8;'>
            왼쪽 사이드바에서 원하는 페이지를 선택하여
            다양한 분석 결과를 확인하실 수 있습니다.
        </p>
        <p style='font-size: 0.95rem; line-height: 1.8;'>
            각 메뉴에서는 상세한 데이터 분석과
            시각화 자료를 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>📧 문의하기</h4>
        <p style='color: #6c757d;'>project@example.com</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>📚 데이터 출처</h4>
        <p style='color: #6c757d;'>서울시 열린데이터광장</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>🔄 업데이트</h4>
        <p style='color: #6c757d;'>분기별 업데이트</p>
    </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;'>
    <p>© 2025 서울시 자치구별 매장 폐업 예측 프로젝트. All rights reserved.</p>
    <p>본 프로젝트는 학생용 교육 목적으로 제작되었습니다.</p>
</div>
""", unsafe_allow_html=True)