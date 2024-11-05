import streamlit as st
import datetime as dt
from PortfolioBuilder import PortfolioBuilder
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    st.set_page_config(page_title="Financial Dashboard", page_icon=":dollar:")

    st.title("Portfolio Optimization Dashboard")
    #st.sidebar.success("Select a page above.")

    #st.markdown("## Portfolio Optimization Dashboard")
    github_url = "https://github.com/MisterMandarino"
    st.sidebar.markdown(
        f'<a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`MisterMandarino`</a>',
        unsafe_allow_html=True,
    )
    
    if "stocks" not in st.session_state:
        st.session_state.stocks_list = ["AAPL, NVDA, AMZN, TSLA"]  # Initialize empty list

    default_tickers_str = ", ".join(st.session_state.stocks_list)

    cont1 = st.container(border=True)
    cont1.markdown("### Portfolio Parameters")
    stocks = cont1.text_input("Enter Tickers: (comma separated)", value=default_tickers_str)
    start, end = cont1.columns(2)
    start_date = start.date_input(
        "Start Date",
        max_value=dt.date.today() - dt.timedelta(days=1),
        min_value=dt.date.today() - dt.timedelta(days=1250),
        value=dt.date.today() - dt.timedelta(days=365),
    )
    end_date = end.date_input(
        "End Date",
        max_value=dt.date.today(),
        min_value=start_date + dt.timedelta(days=1),
        value=dt.date.today(),
    )
    col1, col2 = cont1.columns(2)
    optimization_criterion = col1.selectbox(
        "Optimization",
        options=[
            "Maximize Sharpe Ratio",
            "Minimize Volatility",
            "Maximize Sortino Ratio",
            "Minimize Tracking Error",
            "Minimize Conditional Value-at-Risk",
        ],
    )
    riskFreeRate_d = col2.number_input(
        "Risk Free Rate (%)",
        min_value=0.00,
        max_value=100.00,
        step=0.01,
        format="%0.3f",
        value=3.50,
        help = "the interest an investor would expect from an absolutely risk-free investment over a specified period of time (Treasury Bond Yield)"
    )
    calc = cont1.button("Optimize")
    riskFreeRate = riskFreeRate_d / 100

    st.session_state.stocks_list = [s.strip() for s in stocks.split(",")]

    if calc:
        try:
            with st.spinner("Work in progress..."):
                stocks_list = st.session_state.stocks_list
                optimizer = PortfolioBuilder(
                    stocks_list,
                    start_date,
                    end_date,
                    optimization_criterion,
                    riskFreeRate,
                )
                EF_fig, portfolio_weights, ef_summary = optimizer.get_efficient_frontier()
        except Exception as e:
            st.error(str(e))
            return
        
        with st.container(border=True):
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "Summary",
                    "Efficient Frontier",
                    "Metrics",
                    "Risk Analysis",
                ]
            )
            with tab1:
                st.markdown("#### Portfolio Components")
                st.dataframe(optimizer.pool)

                st.markdown("#### Portfolio Distribution")
                info_fig = optimizer.get_sector_and_market_info()
                st.pyplot(info_fig, use_container_width=True)

                if len(stocks_list) > 1:
                    st.markdown("#### Portfolio Correlation")
                    plt.figure(figsize=(10, 5))
                    sns.heatmap(optimizer.returns.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
                    st.pyplot(plt)
            with tab2:
                st.markdown("#### Efficient Frontier Assets")
                st.table(ef_summary)

                st.markdown("#### Efficient Frontier Graph")
                st.pyplot(EF_fig, use_container_width=True)

                st.markdown("#### Optimized Portfolio Allocation")
                allocationCol, chartCol = st.columns(2)
                with allocationCol:
                    st.table(portfolio_weights)
                with chartCol:
                    allocation_fig = plt.figure(figsize=(10, 5))
                    plt.pie(portfolio_weights['Allocation(%)'], labels=portfolio_weights.index, autopct='%1.1f%%', rotatelabels=False, startangle=90)
                    st.pyplot(allocation_fig, use_container_width=True)

                st.markdown("#### Portfolio Cumulative Returns")
                st.markdown(f"**Time Period**: {(end_date - start_date).days} days")
                cumulative_returns_p, fig = optimizer.portfolioReturnsGraph()
                st.markdown(f'**Portfolio Returns**: {round(cumulative_returns_p.values[-1], 2)}% ')
                st.markdown(f'**Portfolio Volatility**: {round(optimizer.annual_optimized_volatility*100, 2)}% ')
                st.markdown(f'**Portfolio Sharpe-Ratio**: {round(optimizer.optimized_sharpe_ratio, 2)}')
                st.markdown(f'**Portfolio Sharpe-Ratio**: {round(optimizer.optimized_sortino_ratio, 2)}')
                st.markdown(f'**Benchmark Returns (SP500)**: {round(optimizer.benchmark_cumulative_p.values[-1][0], 2)}% ')
                st.pyplot(fig)


            with tab3:
                st.markdown("#### Metrics")
                st.table(optimizer.get_metrics())
                #with st.expander("Metric Interpretations:"):
                #    metric_info()

            with tab4:
                st.markdown("#### VaR and CVaR")
                var = optimizer.riskTable()
                st.table(var.reset_index(drop=True))
            #    with st.expander("VaR and CVar Interpretation"):
            #        var_info()
                st.markdown("#### VaR Breaches")
                st.plotly_chart(optimizer.varXReturns())
            

if __name__ == "__main__":
    main()