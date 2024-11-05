## Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib import patheffects
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import streamlit as st

## Utility Libraries
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt

## Financial Libraries
from financetoolkit import Toolkit
import yfinance as yf

class PortfolioBuilder:
    def __init__(self, tickers, start, end, optimization_criterion, risk_free):
        self.start = start
        self.end = end
        self.risk_free = risk_free
        self.criterion = optimization_criterion
        self.tickers = tickers
        self.weights = np.zeros(len(tickers))

        ## Portfolio
        try:
            self.pool = self._build_asset_pool()
            self.data = yf.download(self.tickers, start=self.start, end=self.end)["Adj Close"]
        except:
            raise ValueError("Unable to download data!")

        if len(self.data.columns) != len(self.tickers):
            raise ValueError("Unable to download data!")

        self.returns = self.data.pct_change().dropna()
        self.std_daily_returns = self.returns.std()
        self.mean_daily_returns = self.returns.mean() 
        self.covMatrix = self.returns.cov()

        ## benchmark
        try:
            sp500 = yf.download('SPY', start=self.start, end=self.end)['Adj Close']
        except:
            raise ValueError("Cannot download Market data")
        self.benchmark_returns = sp500.pct_change().dropna()
        self.benchmark_return = float(self.benchmark_returns.mean() * 252)
        self.benchmark_volatility = float(self.benchmark_returns.std() * np.sqrt(252))
        self.benchmark_cumulative = (self.benchmark_returns + 1).cumprod()
        self.benchmark_cumulative_p = 100 * (self.benchmark_cumulative - 1)

    def portfolioPerformance(self, weights):
        returns = (np.sum(self.mean_daily_returns * weights) * 252)  
        std = np.sqrt(np.dot(weights.T, np.dot(self.covMatrix, weights))) * np.sqrt(252)  
        return returns, std

    def sharpe(self, weights):
        pReturns, pStd = self.portfolioPerformance(weights)  
        return (pReturns - self.risk_free) / pStd

    def sortino(self, weights):
        portfolioDailyReturns = np.dot(self.returns, weights)
        downsideChanges = portfolioDailyReturns[portfolioDailyReturns < 0]
        downside_deviation = downsideChanges.std(ddof=1) * np.sqrt(252)
        meanReturns = portfolioDailyReturns.mean() * 252
        sortino_ratio = (meanReturns - self.risk_free) / downside_deviation
        return sortino_ratio

    def portfolioVariance(self, weights):  
        return self.portfolioPerformance(weights)[1]

    def trackingError(self, weights):
        portfolioDailyReturns = np.array(np.dot(self.returns, weights))
        benchmarkReturns = np.array(self.benchmark_returns)
        difference_array = portfolioDailyReturns - benchmarkReturns
        trackingError = difference_array.std(ddof=1) * np.sqrt(252)
        return trackingError

    def conditionalVar(self, weights):
        portfolioDailyReturns = np.array(np.dot(self.returns, weights))
        mu = portfolioDailyReturns.mean()
        sigma = portfolioDailyReturns.std(ddof=1)
        var = mu + sigma * norm.ppf(0.95)
        loss = portfolioDailyReturns[portfolioDailyReturns < -var]
        cvar = np.mean(loss)
        return -cvar

    def get_sector_and_market_info(self,):
        # Count the occurrences of each sector
        sector_counts = self.pool['Sector'].value_counts()
        # Count the occurrences of each country
        country_counts = self.pool['Country'].value_counts()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Create a pie chart
        axs[0].pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', rotatelabels=False, startangle=90)
        axs[0].set_title('Sector Distribution')
        axs[0].axis('equal')
        axs[1].pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', rotatelabels=False, startangle=90)
        axs[1].set_title('Country Distribution')
        axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Adjust spacing between the plots
        plt.tight_layout()
        return fig
    
    def get_efficient_frontier(self, portfolio_generations=3000):
        # To save all weightage of all portfolios
        all_weights = np.zeros((portfolio_generations, len(self.tickers)))
        results = np.zeros((portfolio_generations, 6))
        
        for i in range(portfolio_generations):
            # Random Weights 
            weights = np.random.random(len(self.tickers))
            weights /= weights.sum()
            # Save weights
            all_weights[i,:] = weights
            # calculate Portfolio Returns
            results[i,0] = self.portfolioPerformance(weights=weights)[0]
            # calculate Portfolio Volatility
            results[i,1] = self.portfolioVariance(weights=weights)
            # calculate Sharpe Ratio
            results[i,2] = self.sharpe(weights=weights)
            # calculate Sortino Ratio
            results[i,3] = self.sortino(weights=weights)
            # calculate Tracking error
            results[i,4] = self.trackingError(weights=weights)
            # calculate Conditional Value-at-Risk
            results[i,5] = self.conditionalVar(weights=weights)

        results = pd.DataFrame(results, columns=['Returns', 'Volatility', 'Sharpe_Ratio', 'Sortino_Ratio', 'Tracking_Error', 'CVAR'])

        if self.criterion == "Maximize Sharpe Ratio":
            ## Highest sharpe ratio
            idx = results.Sharpe_Ratio.idxmax()
        elif self.criterion == "Minimize Volatility":
            ## Lowest volatility
            idx = results.Volatility.idxmin()
        elif self.criterion == "Maximize Sortino Ratio":
            ## Highest sortino ratio
            idx = results.Sortino_Ratio.idxmax()
        elif self.criterion == "Minimize Tracking Error":
            ## Lowest Tracking Error
            idx = results.Tracking_Error.idxmin()
        elif self.criterion == "Minimize Conditional Value-at-Risk":
            ## Lowest Conditional Value-at-Risk
            idx = results.CVAR.idxmin()
        self.annual_optimized_volatility = results.Volatility[idx]
        self.annual_optimized_returns = results.Returns[idx]
        self.optimized_weights = all_weights[idx,:]
        self.optimized_sharpe_ratio = self.sharpe(self.optimized_weights)
        self.optimized_sortino_ratio = self.sortino(self.optimized_weights)

        portfolio_weights = pd.DataFrame(self.optimized_weights*100, index=self.tickers)
        portfolio_weights.columns = ['Allocation(%)']

        ## Summary Stats
        ExpectedReturn = [f"{round(i*252*100, 2)} %" for i in self.mean_daily_returns]
        StandardDeviation = [f"{round(i*np.sqrt(252)*100, 2)} %" for i in self.std_daily_returns]
        sharpeRatio = []
        for i, ret in enumerate(self.mean_daily_returns):
            sharpe = (ret * 252 - self.risk_free) / (self.std_daily_returns[i] * np.sqrt(252))
            sharpeRatio.append(round(sharpe, 2))
        summary = pd.DataFrame(
            {
                "Tickers": self.tickers,
                "Expected Return": ExpectedReturn,
                "Standard Deviation": StandardDeviation,
                "Sharpe Ratio": sharpeRatio,
            }
        )

        fig = plt.figure(figsize=(12,8))
        plt.scatter(results.Volatility, results.Returns, c=results.Sharpe_Ratio, cmap='Blues')
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel('Volatility')
        plt.ylabel('Returns')
        # Best Portfolio
        plt.scatter(self.annual_optimized_volatility, self.annual_optimized_returns, c='red', s=200, marker='*')
        return fig, portfolio_weights, summary
    
    def portfolioReturnsGraph(self):
        portfolio_returns = self.returns.dot(self.optimized_weights)
        portfolio_cumulative_returns = (1+portfolio_returns).cumprod()
        portfolio_cumulative_returns_p = 100 * (portfolio_cumulative_returns - 1)

        fig = plt.figure(figsize=(12,7))
        plt.plot(portfolio_cumulative_returns_p.index, portfolio_cumulative_returns_p, label='Portfolio', color='blue')
        plt.plot(self.benchmark_cumulative_p.index, self.benchmark_cumulative_p, label='Market (SP500)', color='orange')
        plt.legend()
        return portfolio_cumulative_returns_p, fig

    def MMeanReturn(self, frequency):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        if frequency == "monthly":
            return portfolioDailyReturns.mean() * 21 * 100
        if frequency == "annual":
            return portfolioDailyReturns.mean() * 252 * 100

    def MStandardDeviation(self, frequency):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        if frequency == "monthly":
            return portfolioDailyReturns.std(ddof=1) * np.sqrt(21) * 100
        if frequency == "annual":
            return portfolioDailyReturns.std(ddof=1) * np.sqrt(252) * 100

    def MDownsideDeviation(self):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        downsideChanges = portfolioDailyReturns[portfolioDailyReturns < 0]
        return downsideChanges.std(ddof=1) * np.sqrt(252) * 100

    def MMaxDrawdown(self):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        returns = np.array(portfolioDailyReturns)
        cumulative_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) * 100

        ## Fix this calculation

        return max_drawdown

    def MBeta(self):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        portfolio = np.array(portfolioDailyReturns).flatten()
        benchmark = np.array(self.benchmark_returns).flatten()
        portfolio_dates = pd.date_range(
            start=self.start, periods=len(portfolio), freq="D"
        )
        benchmark_dates = pd.date_range(
            start=self.start, periods=len(benchmark), freq="D"
        )

        portfolio_series = pd.Series(portfolio, index=portfolio_dates)
        benchmark_series = pd.Series(benchmark, index=benchmark_dates)

        common_dates = portfolio_series.index.intersection(benchmark_series.index)

        aligned_portfolio = portfolio_series[common_dates]
        aligned_benchmark = benchmark_series[common_dates]

        portfolio = aligned_portfolio.tolist()
        benchmark = aligned_benchmark.tolist()

        returns_data = pd.DataFrame(
            {"Portfolio": portfolio, "Benchmark": benchmark}
        ).dropna()
        X = sm.add_constant(returns_data["Benchmark"])
        model = sm.OLS(returns_data["Portfolio"], X).fit()
        beta = model.params["Benchmark"]

        return beta

    def MAlpha(self):
        market_return = float(self.benchmark_returns.mean() * 252)
        annual_return = self.MMeanReturn("annual") / 100
        alpha = (annual_return) - (
            self.risk_free + self.MBeta() * (market_return - self.risk_free)
        )
        return alpha * 100

    def MSharpeRatio(self):
        annual_std = self.MStandardDeviation("annual") / 100
        annual_return = self.MMeanReturn("annual") / 100
        sharpe = (annual_return - self.risk_free) / annual_std
        ## fix sharpe ratio
        return sharpe

    def MSortinoRatio(self):
        downside_std = self.MDownsideDeviation() / 100
        annual_return = self.MMeanReturn("annual") / 100
        sortino = (annual_return - self.risk_free) / downside_std
        return sortino

    def MTrackingError(self):
        portfolioDailyReturns = np.array(np.dot(self.returns, self.optimized_weights))
        benchmarkReturns = np.array(self.benchmark_returns)
        difference_array = portfolioDailyReturns - benchmarkReturns
        trackingError = difference_array.std(ddof=1) * np.sqrt(252)
        return trackingError

    def MPositivePeriods(self):
        portfolioDailyReturns = np.dot(self.returns, self.optimized_weights)
        positive = portfolioDailyReturns[portfolioDailyReturns > 0]
        total = len(portfolioDailyReturns)
        positive_periods = len(positive)
        ratio = round((positive_periods / (total)) * 100, 2)
        return f"{positive_periods} out of {total} ({ratio}%)"
    
    def get_metrics(self,):
        metric_df = {
                    "Mean Return (Monthly)": f'{round(self.MMeanReturn("monthly"), 2)}%',
                    "Mean Return (Annualised)": f'{round(self.MMeanReturn("annual"), 2)}%',
                    "Standard Deviation (Monthly)": f'{round(self.MStandardDeviation("monthly"), 2)}%',
                    "Standard Deviation (Annualised)": f'{round(self.MStandardDeviation("annual"), 2)}%',
                    "Downside Standard Deviation": f"{round(self.MDownsideDeviation(), 2)}%",
                    "Maximum Drawdown": f"{round(self.MMaxDrawdown(), 2)}%",
                    "Beta": round(self.MBeta(), 2),
                    "Alpha": f"{round(self.MAlpha(), 2)}%",
                    "Sharpe Ratio": round(self.MSharpeRatio(), 2),
                    "Sortino Ratio": round(self.MSortinoRatio(), 2),
                    "Tracking Error": round(self.MTrackingError(), 2),
                    "Positive Periods": self.MPositivePeriods(),
                }
        return metric_df

    def Rvar(self):
        portfolio_daily = np.dot(self.returns, self.optimized_weights)
        mu, sigma = portfolio_daily.mean(), portfolio_daily.std(ddof=1)
        confidence_levels = [0.9, 0.95, 0.99]
        var_values = []
        for lvl in confidence_levels:
            var = mu + sigma * norm.ppf(lvl)
            var_values.append(var)
        return var_values

    def RCvar(self):
        portfolio_daily = np.dot(self.returns, self.optimized_weights)
        var_values = self.Rvar()
        cvar_values = []
        for var in var_values:
            loss = portfolio_daily[portfolio_daily < -var]
            cvar = np.mean(loss)
            cvar_values.append(-cvar)
        return cvar_values

    def riskTable(self):
        var_values = self.Rvar()
        cvar_values = self.RCvar()

        var_dict = {
            "90%": [
                "90%",
                f"{round(var_values[0]*100, 2)}%",
                f"{round(cvar_values[0]*100, 2)}%",
            ],
            "95%": [
                "95%",
                f"{round(var_values[1]*100, 2)}%",
                f"{round(cvar_values[1]*100, 2)}%",
            ],
            "99%": [
                "99%",
                f"{round(var_values[2]*100, 2)}%",
                f"{round(cvar_values[2]*100, 2)}%",
            ],
        }

        var_df = pd.DataFrame(
            var_dict, index=["Confidence Level", "VaR(%)", "CVaR(%)"]
        ).T

        return var_df

    def varXReturns(self):
        portfolio_daily = np.dot(self.returns, self.optimized_weights)
        portfolio = np.array(portfolio_daily).flatten()
        portfolio_dates = self.returns.index

        if len(portfolio) == len(portfolio_dates):
            portfolio_series = pd.Series(portfolio, index=portfolio_dates)
        else:
            # Handle length mismatch: truncate or pad the shorter array
            min_length = min(len(portfolio), len(portfolio_dates))
            portfolio_series = pd.Series(
                portfolio[:min_length], index=portfolio_dates[:min_length]
            )

        daily_returns_df = pd.DataFrame(
            {
                "Date": portfolio_series.index,
                "Daily Return (%)": portfolio_series.values * 100,
            }
        )

        ymin = np.min(portfolio_series.values * 100)
        ymax = np.max(portfolio_series.values * 100)

        var_list = self.Rvar()
        var_95 = var_list[1]

        breach_points = daily_returns_df[
            daily_returns_df["Daily Return (%)"] < -var_95 * 100
        ]
        fig = px.line(daily_returns_df, "Date", "Daily Return (%)")
        scatter = fig.add_scatter(
            x=breach_points["Date"],
            y=breach_points["Daily Return (%)"],
            mode="markers",
            marker_color="#0D2A63",
            name="VaR 95% Breaches",
        )

        fig.update_yaxes(tickformat=".0f", ticksuffix="%")
        fig.update_yaxes(range=[ymin * 1.4, ymax * 1.4])
        fig.update_traces(line_color="#86caff")

        fig.update_layout(
            legend_title_text="",
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
        )
        return fig

    def _build_asset_pool(self,):
        portfolio_pool = []
        for ticker in self.tickers:
            try:
                asset = yf.Ticker(ticker)
            except:
                print(f'Error: could not download data for {ticker}')
                continue
            data = {}
            data['Company'] = asset.info['longName']
            data['Ticker'] = ticker
            data['Industry'] = asset.info['industry']
            data['Sector'] = asset.info['sector']
            data['Country'] = asset.info['country']
            data['Price'] = asset.info['currentPrice']
            portfolio_pool.append(data)
        return pd.DataFrame(portfolio_pool)
    
class Analyzer:
    def __init__(self, ticker, start, api_key=""):
        # Initialize the Toolkit with company tickers
        self.ticker = ticker
        try:
            self.asset = Toolkit(ticker, api_key=api_key, start_date=start)
            self.asset_profile = self.asset.get_profile()
            self.income_statement = self.asset.get_income_statement()
            self.cash_flow = self.asset.get_cash_flow_statement()
            self.balance_sheet = self.asset.get_balance_sheet_statement()
            self.treasury = self.asset.get_treasury_data()
            self.profitability_ratios = self.asset.ratios.collect_profitability_ratios()
            self.esg_scores = self.asset.get_esg_scores()
        except Exception as e:
            print(e)

    def plot_cumulative_returns(self, period='daily'):
        company_name = self.asset_profile.loc["Company Name"]
        # Create a line chart for cumulative returns
        ax = self.asset.get_historical_data(period=period)["Cumulative Return"].plot(figsize=(15, 5),lw=2,)
        # Customize the colors and line styles
        ax.set_prop_cycle(color=["#007ACC", "#FF6F61"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Return")
        ax.set_title(f"Cumulative Returns of {company_name} and the S&P 500")
        # Add a legend
        ax.legend([self.ticker, "S&P 500"], loc="upper left")
        # Add grid lines for clarity
        ax.grid(True, linestyle="--", alpha=0.7)
        return ax
    
    def plot_reinvestment_information(self,):
        net_income = self.cash_flow.loc["Net Income"] / 1_000_000
        capex = -(self.cash_flow.loc["Capital Expenditure"] / 1_000_000)
        retained_earnings = self.balance_sheet.loc["Retained Earnings"] / 1_000_000
        combined_df = pd.concat([net_income, capex, retained_earnings], axis=1)
        melted_df = combined_df.reset_index().melt(id_vars='date', var_name='Type', value_name='Amount')
        print(melted_df)
        # Set up the plot
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        # Plot the profits and reinvestments
        sns.barplot(x='date', y='Amount', hue='Type', data=melted_df)
        # Customize the plot
        plt.title("Company Profits and Reinvestments Over Time")
        plt.ylabel("Amount (in thousands)")
        plt.xlabel("Year")
        plt.legend(title="Type")
        # Show the plot
        return fig
    
    def plot_EBITDA(self, ):
        # Create the bar plot
        plt.clf()
        ebitda_data = self.income_statement.loc["EBITDA", :].T
        colors = ["#007ACC"]
        ax = ebitda_data.plot.bar(figsize=(15, 5), color=colors)
        # Add data labels on top of the bars with custom formatting (divided by 1,000,000 for millions and thousand separator)
        for p in ax.patches:
            ebitda_millions = p.get_height() / 1_000_000
            label = f"{ebitda_millions:,.2f} M"
            x = p.get_x() + p.get_width() / 2.0
            y = p.get_height()
            # Check if the label is too close to the top of the chart
            if y < 0.2 * ax.get_ylim()[1]:
                va = "bottom"
                xytext = (0, 5)
            else:
                va = "top"
                xytext = (0, -5)
            # Create a stroke effect for the text
            text = ax.annotate(
                label,
                (x, y),
                ha="center",
                va=va,
                fontsize=10,
                color="black",
                xytext=xytext,
                textcoords="offset points",
            )
            text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
        # Customize the axis labels
        plt.xlabel("", fontsize=10)
        plt.ylabel("", fontsize=10)
        # Change the degree of the x ticks
        plt.xticks(rotation=0)
        # Customize the title
        plt.title(f"Earnings Before Interest, Taxes, Depreciation and Amortization (EBITDA) of {self.ticker}",fontsize=12,)
        # Add a horizontal grid for clarity
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # Add a thousand separator to the y-axis
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1_000_000:,.0f} M"))
        # Show the plot
        st.pyplot(plt)

    def plot_profitability_ratios(self, ):
        plt.clf()
        ratios_to_plot = [
            "Return on Assets",
            "Return on Equity",
            "Return on Invested Capital",
            "Return on Tangible Assets",
        ]
        # Create the plot
        ax = (
            (self.profitability_ratios.dropna(axis=1) * 100)
            .loc[ratios_to_plot, :]
            .T.plot(figsize=(15, 5), title=f"Profitability Ratios for {self.ticker}", lw=2)
        )
        # Customize the line styles and colors
        line_styles = ["-", "--", "-.", ":"]
        line_colors = ["blue", "red", "green", "purple"]
        for i, line in enumerate(ax.get_lines()):
            line.set_linestyle(line_styles[i])
            line.set_color(line_colors[i])
        # Customize the legend
        ax.legend(ratios_to_plot)
        # Add labels and grid
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        # Customize the title
        plt.title("Profitability Ratios")
        # Show the plot
        st.pyplot(plt)

    def plot_value_at_risk(self, period="weekly"):
        plt.clf()
        value_at_risk = self.asset.risk.get_value_at_risk(period=period)
        # Filter out the occasional positive return
        value_at_risk = value_at_risk[value_at_risk < 0]
        # Create an area chart for Value at Risk (VaR) with custom styling
        fig, ax = plt.subplots(figsize=(15, 5))
        # Customize the colors and transparency
        ax.set_prop_cycle(color=["#007ACC", "#FF6F61"])
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("VaR", fontsize=12)
        # Add a title with a unique font style
        ax.set_title("Value at Risk (VaR)")
        # Stack the area chart for a better visual representation
        value_at_risk.plot.area(ax=ax, stacked=False, alpha=0.7)
        # Add grid lines with a unique linestyle
        ax.grid(True, linestyle="--", alpha=0.5)
        # Customize the ticks and labels with a unique font
        ax.tick_params(axis="both", which="major", labelsize=10)
        # Add a background color to the plot area for a unique touch
        ax.set_facecolor("#F0F0F0")
        # Add a unique border color to the plot area
        ax.spines["top"].set_color("#E0E0E0")
        ax.spines["bottom"].set_color("#E0E0E0")
        ax.spines["left"].set_color("#E0E0E0")
        ax.spines["right"].set_color("#E0E0E0")
        # Set a unique background color for the entire plot
        fig.set_facecolor("#F8F8F8")
        # Display the prettier VaR chart with the annotation
        st.pyplot(plt)

    def plot_ESG_score(self,):
        plt.clf()
        score = self.esg_scores.xs(self.ticker, level=1, axis=1)
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.2  # Width of each bar
        # Create positions for each quarter
        quarters = range(len(score['date']))
        # Plot each ESG score as a bar series
        ax.bar([x - width for x in quarters], score['Environmental Score'], width=width, label='Environmental', color='skyblue')
        ax.bar(quarters, score['Social Score'], width=width, label='Social', color='salmon')
        ax.bar([x + width for x in quarters], score['Governance Score'], width=width, label='Governance', color='lightgreen')
        # Plot the average ESG line
        ax.plot(quarters, score['ESG Score'], color='black', marker='o', linestyle='-', linewidth=2, label='ESG score')
        # Labeling
        ax.set_xticks(quarters)
        ax.set_xticklabels(score['date'])
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Score')
        ax.set_title('Quarterly ESG Scores')
        ax.legend()
        st.pyplot(plt)

    def plot_factors_correlation(self, period='quaterly'):
        plt.clf()
        factor_asset_correlations = self.asset.performance.get_factor_asset_correlations(period=period)
        # Define your factor_asset_correlations DataFrame (replace YourDataFrame)
        correlations = factor_asset_correlations.xs(self.ticker, axis=1, level=0)
        # Create subplots with shared x-axis and customize styles
        fig, ax = plt.figure(figsize=(15, 8))
        # Set a color palette for the lines
        colors = ["#007ACC", "#FF6F61", "#4CAF50", "#FFD700", "#FF6347", "#6A5ACD", "#FF8C00"]
        # Plot correlations 
        correlations.plot(ax=ax, title=f"Correlations of {self.ticker} with the Fama-French Factors", color=colors)
        ax.set_ylabel("Correlation")
        ax.set_xlabel("Date")
        # Add grid lines for clarity
        ax.grid(True, linestyle="--", alpha=0.5)
        # Customize the legend and labels
        ax.legend(loc="upper right", frameon=False)
        # Set background colors for the subplots
        ax.set_facecolor("#F9F9F9")
        # Remove spines on the top and right side of the subplots
        ax.spines[["top", "right"]].set_visible(False)
        # Show the plots
        st.pyplot(plt)