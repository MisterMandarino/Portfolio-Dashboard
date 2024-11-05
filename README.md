# Portfolio Dashboard
The dashboard is built using <a href="https://streamlit.io/" target="_blank">Streamlit</a> and the<a href="https://github.com/JerBouma/FinanceToolkit" target="_blank">FinanceToolkit</a>

To be able to get started, you need to obtain an API Key from FinancialModelingPrep. This is used to gain access to 30+ years of financial statement both annually and quarterly. Note that the Free plan is limited to 250 requests each day, 5 years of data and only features companies listed on US exchanges.
___ 

<b><div align="center">Obtain an API Key from the <a href="https://site.financialmodelingprep.com/" target="_blank">FinancialModelingPrep</a>Website.</div></b>
___

# Installation

1. To avoid possible conflicts, it is recommended to use a **virtual environment** to install the required packages. 
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ``` 
    In Alternative (windows)
    ```
    python3 -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
2. To install the project, clone the repository and install the required packages with the following command:
    ```
    pip install -r requirements.txt
    ```
3. Run the streamlit application
    ```
    streamlit run PortfolioOptimizer.py
    ```