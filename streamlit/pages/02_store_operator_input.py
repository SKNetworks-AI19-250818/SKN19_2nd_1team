import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(layout="wide")  # ← 새로 추가 (파일의 첫 Streamlit 호출이어야 함)

# (중요) 멀티페이지에서는 app.py 한 곳에서만 set_page_config를 호출하세요.
# st.set_page_config(...)  # ← 이 줄은 삭제/주석 처리


st.title("🏪 매장 운영자 정보 입력")  # 최소 렌더 보장

# ───────────────────────────────
# CSS (홈 톤 유지)
# ───────────────────────────────
try:
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 10px; margin-bottom: 2rem;
            color: white; text-align: center;
        }
        .input-box, .stat-card {
            background: white; padding: 1.5rem; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem;
        }
        .stat-card { text-align: center; transition: transform 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .stat-value { font-size: 2rem; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 1rem; color: #6c757d; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container">
        <h2 style='margin:0;'>매장 운영자 정보 입력</h2>
        <p style='margin-top: .5rem; opacity: .9'>현재 운영 중인 매장의 정보를 입력하면 폐업 위험도를 알려드립니다.</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error("렌더링 중 오류가 발생했습니다.")
    st.exception(e)
    con
# ───────────────────────────────
# 입력 폼
# ───────────────────────────────
with st.form("store_form"):
    st.markdown("<div class='input-box'><h4>📍 기본 정보</h4></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gu_name = st.selectbox(
            "자치구 코드 명",
            ['강남구','강동구','강북구','강서구','관악구','광진구','구로구','금천구',
             '노원구','도봉구','동대문구','동작구','마포구','서대문구','서초구',
             '성동구','성북구','송파구','양천구','영등포구','용산구','은평구','종로구','중구','중랑구']
        )
    with c2:
        svc_name = st.selectbox(
            "서비스 업종 코드 명",
            ['편의점','카페','음식점','미용실','세탁소','PC방','노래방','기타']
        )

    st.markdown("<div class='input-box'><h4>💰 매출 및 임대료</h4></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        prev_sales = st.number_input("이전 분기 당월 매출 (원)", min_value=0, step=10000)
        rent = st.number_input("이전 분기 임대료 (원)", min_value=0, step=10000)
    with c4:
        prev_sales_cnt = st.number_input("이전 분기 당월 매출 건수 (건)", min_value=0, step=1)
        weekend_ratio = st.slider("주말/야간 매출 비율 (%)", 0, 100, 30)

    # ✅ 버튼은 '반드시' 폼 안의 마지막에 배치
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        submitted = st.form_submit_button("📊 통계 보기", use_container_width=True)
    with col_btn2:
        result_btn = st.form_submit_button("🔎 결과 확인", use_container_width=True)

# ------------------------------------------
# 결과 확인 모달(가상 데이터, UI 미리보기)
# ------------------------------------------
# 세션키 초기화 (첫 로드 KeyError 방지)
if "show_result_modal" not in st.session_state:
    st.session_state["show_result_modal"] = False

# 버튼이 눌린 그 순간에만 True로 세팅 (submit은 rerun을 유발)
if result_btn:
    st.session_state["show_result_modal"] = True

# 모달 렌더는 '조건 안에서만'
if st.session_state.get("show_result_modal"):
    fake_risk = 6.1
    fake_group = "중위 위험 구간 (Q3)"
    fake_msg = f"{gu_name} · {svc_name} 기준, 최근 분기 입력치를 바탕으로 산출한 가상 예측입니다."

    MODAL_HTML = f"""
    <!-- (여기에 네가 작성한 모달 CSS/HTML 전체 그대로) -->
    """

    components.html(MODAL_HTML, height=720, scrolling=False)
# ───────────────────────────────
# 결과
# ───────────────────────────────
if submitted:
    st.markdown("---")
    st.subheader("📈 입력 정보 기반 통계 요약")

    avg_sales = prev_sales / prev_sales_cnt if prev_sales_cnt > 0 else 0
    rent_ratio = (rent / prev_sales * 100) if prev_sales > 0 else 0
    weekend_sales_est = prev_sales * (weekend_ratio / 100)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">건당 평균 매출액</div>
            <div class="stat-value">{avg_sales:,.0f}원</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">임대료 / 매출 비율</div>
            <div class="stat-value">{rent_ratio:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">주말/야간 매출 추정액</div>
            <div class="stat-value">{weekend_sales_est:,.0f}원</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"💡 {gu_name} {svc_name} 업종 기준. 추후 예측 모델을 연동합니다.")
