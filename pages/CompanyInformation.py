import streamlit as st
import datetime as dt
from PortfolioBuilder import Analyzer

st.title("Asset Analysis")

github_url = "https://github.com/MisterMandarino"
st.sidebar.markdown(
    f'<a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`MisterMandarino`:tangerine:</a>',
    unsafe_allow_html=True,
)

cont1 = st.container(border=True)
cont1.markdown("### Asset Parameters")
fmp_url = "https://site.financialmodelingprep.com/pricing-plans"
cont1.markdown("Get your FinancialModelingPrep API KEY [Here](%s)" % fmp_url)
API_KEY = cont1.text_input("Enter API KEY: ", value="", help="To be able to get started, you need to obtain an API Key from FinancialModelingPrep. If not provided the data will be retrieved from Yahoo Finance.")
ticker_column, start_column = cont1.columns(2)
ticker = ticker_column.text_input("Ticker", value="")
start_date = str(start_column.date_input(
    "Start Date",
    max_value=dt.date.today() - dt.timedelta(days=1),
    min_value=dt.date.today() - dt.timedelta(days=1250),
    value=dt.date.today() - dt.timedelta(days=365),
))
#start_date = start_date.strftime("%Y/%m/%d")

analysis_button = cont1.button("Get Info")

if analysis_button:
    try:
        with st.spinner("Work in progress..."):
            analysis = Analyzer(ticker=ticker, start=start_date, api_key=API_KEY)
    except Exception as e:
        st.error(str(e))
    
    with st.container(border=True):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Profile",
                "Balance Sheet",
                "Income Statement",
                "Cash Flow",
                #"ESG Score",
                "Market Factors (CAPM)",
                "Economics"
            ]
        )
        with tab1:
            st.markdown("#### Company Profile")
            st.table(analysis.asset_profile)
            st.markdown("#### Cumulative Returns")
            analysis.plot_cumulative_returns()
        with tab2:
            st.markdown("#### Balance Sheet")
            st.table(analysis.balance_sheet)
            st.markdown("#### Company Income and Investment")
            analysis.plot_reinvestment_information()
        with tab3:
            st.markdown("#### Income Statement")
            st.table(analysis.income_statement)
            st.markdown("#### Company EBITDA")
            analysis.plot_EBITDA()
        with tab4:
            st.markdown("#### Cash Flow")
            st.table(analysis.cash_flow)
            st.markdown("#### Profitability Ratios")
            analysis.plot_profitability_ratios()
            st.markdown("#### Value at Risk (VAR)")
            analysis.plot_value_at_risk()
        #with tab5:
            #st.markdown("#### Environmental, Social, Governance Score (ESG)")
            #st.table(analysis.esg_scores)
            #analysis.plot_ESG_score()
        with tab5:
            st.markdown("#### Market Factors Correlation")
            analysis.plot_factors_correlation()
            st.markdown("#### Market Factors Sensitivity (R-squared)")
            analysis.plot_factors_sensitivity()
        with tab6:
            st.markdown("#### US Treasury Yield Curve")
            analysis.plot_US_treasury_yield()
            st.markdown("#### House Price Index")
            analysis.plot_house_price_index()
            #st.markdown("#### Unemployment Rates")
            #analysis.plot_unemployment_rate()