# -*- coding: UTF-8 -*-
#
# * ---------------------------------------------------------------- *
# *
# * Project Description:
# *
# * ---DATE--   --DESCRIPTION------------------------------ -AUTHOR- *
# * 06Sep2025   Creation                                     JFRD
# * ---------------------------------------------------------------- *
import sys
import time
import argparse
import os
import signal

import logging
from icecream import ic

ic.configureOutput(includeContext=True)
# ic.disable()

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import yfinance as yf

import simfin as sf

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import xgboost as xgb

# * --------------------- *
# *    Globals (Begin)    *
# * --------------------- *

REGRESSOR_LIST = [
    "DecisionTreeRegressor",
    "ElasticNet",
    "KNeighborsRegressor",
    "LinearRegression",
    "SVR",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBRegressor",
]

# * --------------------- *
# *    Globals (End)      *
# * --------------------- *


def getRegressorList():
    return REGRESSOR_LIST


def getStatement(localPath, stmt="income", variant="annual"):
    if stmt not in ["cashflow", "income", "balance", "shareprices"]:
        raise Exception("Invalid Statement")

    data_file_name = localPath + "us-" + stmt + "-" + variant + ".csv"

    if os.path.exists(data_file_name):
        if stmt == "shareprices":
            df = pd.read_csv(data_file_name, sep=";", parse_dates=["Date"])
            df.drop(["Dividend", "Shares Outstanding"], axis=1, inplace=True)
            df.dropna(inplace=True)
        else:
            df = pd.read_csv(data_file_name, sep=";")
        # print(df.head())
        return df

    # Set your API-key for downloading data.
    SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")
    sf.set_api_key(SIMFIN_API_KEY)

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir(localPath)

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    if stmt == "cashflow":
        df = sf.load_cashflow(variant="quarterly")
        df = sf.load_cashflow(variant=variant)
    elif stmt == "income":
        df = sf.load_income(variant="quarterly")
        df = sf.load_income(variant=variant)
    elif stmt == "balance":
        df = sf.load_balance(variant="quarterly")
        df = sf.load_balance(variant=variant)
    elif stmt == "shareprices":
        df = sf.load_shareprices(variant=variant)

    df.reset_index(inplace=True)

    if stmt == "shareprices":
        df.drop(["Dividend", "Shares Outstanding"], axis=1, inplace=True)
        df.dropna(inplace=True)

    # Print the first rows of the data.
    # print(df.head())
    # print(df.columns)

    return df


def getXDataMerged(localPath):
    """
    Para combinar datos financieros fundamentales de SimFin o SimFin+ (https://simfin.com/) sin API.
    Descargar archivos de Estado de Resultados, Balance General y Flujo de Caja,
    Colóquelo en un directorio y proporcione la ruta del directorio a la función. Asume nombres de archivo estándar de SimFin.
    Devuelve un DataFrame del resultado combinado. Imprime información de archivo.
    """

    incomeStatementData = getStatement(localPath, stmt="income")
    balanceSheetData = getStatement(localPath, stmt="balance")
    CashflowData = getStatement(localPath, stmt="cashflow")

    print(
        "Los datos CSV del estado de resultados son(filas, columnas): ",
        incomeStatementData.shape,
    )
    print("Los datos CSV del balance general son: ", balanceSheetData.shape)
    print("Los datos CSV de flujo de efectivo son: ", CashflowData.shape)

    # Merge the data together
    result = pd.merge(
        incomeStatementData,
        balanceSheetData,
        on=[
            "Ticker",
            "SimFinId",
            "Currency",
            "Fiscal Year",
            # "Report Date",
            "Publish Date",
        ],
    )
    result = pd.merge(
        result,
        CashflowData,
        on=[
            "Ticker",
            "SimFinId",
            "Currency",
            "Fiscal Year",
            # "Report Date",
            "Publish Date",
        ],
    )

    # dates in correct format
    result["Report Date"] = pd.to_datetime(result["Report Date"])
    result["Publish Date"] = pd.to_datetime(result["Publish Date"])
    print("Merged X data matrix shape is: ", result.shape)
    return result


def getYRawData(localPath):
    """
    Read stock price data from SimFin or SimFin+ (https://simfin.com/),
    without API.
    Place in a directory and give the directory path to the function.
    Assumes standard filenames from SimFin.
    Returns a DataFrame.
    Prints file info.
    """
    dailySharePrices = getStatement(localPath, stmt="shareprices", variant="daily")
    dailySharePrices["Date"] = pd.to_datetime(dailySharePrices["Date"])
    print("Stock Price data matrix is: ", dailySharePrices.shape)
    return dailySharePrices


def getYPriceDataNearDate(ticker, date, modifier, dailySharePrices):
    windowDays = 5

    dd = pd.to_datetime(date)
    ddi = dd + pd.Timedelta(days=modifier)
    ddf = dd + pd.Timedelta(days=(modifier + windowDays))

    try:
        rows = dailySharePrices.loc[(ticker, ddi):(ticker, ddf)]
        # rows = dailySharePrices.loc[ticker][ddi:ddf]
    except KeyError:
        return [ticker, float(np.nan), np.datetime64("NaT"), float(np.nan)]

    if rows.empty:
        return [ticker, float(np.nan), np.datetime64("NaT"), float(np.nan)]
    else:
        return [
            ticker,
            rows.iloc[0]["Open"],
            rows.iloc[0]["Date"],
            rows.iloc[0]["Volume"] * rows.iloc[0]["Open"],
        ]


def getYPricesReportDateAndTargetDate(x, d, modifier=365):
    """
    Takes in all fundamental data X, all stock prices over time y,
    and modifier (days), and returns the stock price info for the
    data report date, as well as the stock price one year from that date
    (if modifier is left as modifier=365)
    """
    # Preallocation list of list of 2
    # [(price at date) (price at date + modifier)]
    y = [[None] * 8 for i in range(len(x))]
    whichDateCol = "Publish Date"  # or 'Report Date',
    # is the performance date from->to. Want this to be publish date.
    # Because of time lag between report date
    # (which can't be actioned on) and publish date
    # (data we can trade with)
    i = 0
    for index in tqdm(range(len(x)), leave=False):
        ticker = x["Ticker"].iloc[index]
        fromDate = x[whichDateCol].iloc[index]
        y[i] = getYPriceDataNearDate(
            ticker,
            fromDate,
            0,
            d,
        ) + getYPriceDataNearDate(
            ticker,
            fromDate,
            modifier,
            d,
        )
        i = i + 1

    return y


def getStockPricesIndexed(localPath):
    d = getYRawData(localPath)
    d.reset_index(drop=True, inplace=True)
    d.set_index(["Ticker", "Date"], inplace=True, drop=False)
    return d


def getY(localPath, X):
    d = getStockPricesIndexed(localPath)

    print(getYPriceDataNearDate("GOOG", "2019-09-23", 0, d))
    print(getYPriceDataNearDate("AAPL", "2019-10-12", 30, d))

    # We want to know the performance for each stock, each year, between 10-K report dates.
    # takes VERY long time, several hours.
    # because of lookups in this function.
    y = getYPricesReportDateAndTargetDate(X, d, 365)
    return y


def calcZScores(X):
    """
    Calculate Altman Z'' scores 1995
    """
    Z = pd.DataFrame()
    Z["Z score"] = (
        3.25
        + 6.51 * X["(CA-CL)/TA"]
        + 3.26 * X["RE/TA"]
        + 6.72 * X["EBIT/TA"]
        + 1.05 * X["Book Equity/TL"]
    )
    return Z


def getStockPriceBetweenDates(date1, date2, ticker, d):
    rows = d.loc[(ticker, date1):(ticker, date2)]

    return rows


def getStockPriceData(
    dateTimeIndex, ticker, y_withData, mask, daily_stock_prices, rows
):
    """
    Get the stock price for a ticker
    between the buy/sell date (using y_withdata)
    2021 version change to select from March to March only,
    go for more corresponding backtest to reality,
    rather than attampting to match the training data closely.
    """
    # date1 = y_withData[mask][y_withData[mask]["Ticker"] == ticker]["Date"].values[0]
    # date2 = y_withData[mask][y_withData[mask]["Ticker"] == ticker]["Date2"].values[0]
    date1 = dateTimeIndex[0]
    date2 = dateTimeIndex[-1]
    rows = getStockPriceBetweenDates(date1, date2, ticker, daily_stock_prices)

    return rows


def getDataForDateRange(date_Index_New, rows):
    """
    Given a date range(index), and a series of rows,
    that may not correspond exactly,
    return a DataFrame that gets rows data,
    for each period in the date range(index)
    """
    WeeklyStockDataRows = pd.DataFrame()
    for I in date_Index_New:
        WeeklyStockDataRows = pd.concat(
            [
                WeeklyStockDataRows,
                rows.iloc[
                    rows.index.get_indexer([pd.to_datetime(I)], method="nearest")
                ],
            ],
            ignore_index=True,
        )
    return WeeklyStockDataRows


def getStockTimeSeries(dateTimeIndex, y_withData, tickers, mask, daily_stock_prices):
    """
    Get the stock price as a time series DataFrame
    for a list of tickers.
    A mask is used to only consider stocks for a certain period.
    dateTimeIndex is typically a weekly index,
    so we know what days to fetch the price for.
    """
    stockRet = pd.DataFrame(index=dateTimeIndex)
    dTI_new = dateTimeIndex.strftime("%Y-%m-%d")  # Change Date Format
    rows = pd.DataFrame()
    for tick in tickers:
        # Here "rows" is stock price time series data
        # for individual stock
        rows = getStockPriceData(
            dateTimeIndex, tick, y_withData, mask, daily_stock_prices, rows
        )
        rows.index = pd.DatetimeIndex(rows["Date"])
        WeeklyStockDataRows = getDataForDateRange(dTI_new, rows)
        # Here can use Open, Close, Adj. Close, etc. price
        stockRet[tick] = WeeklyStockDataRows["Close"].values

    return stockRet


def getPortfolioRelativeTimeSeries(stockRet):
    """
    Takes DataFrame of stock returns, one column per stock
    Normalises all the numbers so the price at the start is 1.
    Adds a column for the portfolio value.
    """
    for key in stockRet.keys():
        stockRet[key] = stockRet[key] / stockRet[key].iloc[0]
    stockRet["Portfolio"] = stockRet.sum(axis=1) / (stockRet.keys().shape[0])
    return stockRet


def getTickerPerformance(
    ticker, y_withData, start_date, end_date, daily_stock_prices_data
):
    dateTimeIndex = pd.date_range(start=start_date, end=end_date, freq="W")
    thisYearMask = y_withData["Date"].between(
        pd.to_datetime(start_date),  ######
        pd.to_datetime(end_date),
    )

    stockRet = getStockTimeSeries(
        dateTimeIndex,
        y_withData,
        [ticker],
        thisYearMask,
        daily_stock_prices_data,
    )
    stockRetRel = getPortfolioRelativeTimeSeries(stockRet)

    return stockRetRel


def getPortTimeSeriesForYear(
    date_starting, y_withData, X, daily_stock_prices, ml_model_pipeline
):
    """
    Function runs a backtest.
    Returns DataFrames of selected stocks/portfolio performance,
    for 1 year.
    y_withData is annual stock performances (all backtest years)
    date_starting e.g. '2010-01-01'
    daily_stock_prices is daily(mostly) stock price time series for
    all stocks
    """

    # get y dataframe with ticker performance only
    y = getYPerf(y_withData)

    # Get performance only for time frame we care about,
    # mask original data using the start date
    thisYearMask = y_withData["Date"].between(
        pd.to_datetime(date_starting) - pd.Timedelta(days=120),  ######
        pd.to_datetime(date_starting),
    )

    # Get return prediction from model
    y_pred = ml_model_pipeline.predict(X[thisYearMask])

    # Make it a DataFrame to select the top picks
    y_pred = pd.DataFrame(y_pred)

    ##### Change in code for Z score filtering #####
    # Separate out stocks with low Z scores
    z = calcZScores(X)

    # 3.75 is approx. B- rating
    bl_safeStocks = z["Z score"][thisYearMask].reset_index(drop=True) > 2
    y_pred_z = y_pred[bl_safeStocks]

    # Get bool list of top stocks
    bl_bestStocks = y_pred_z[0] > y_pred_z.nlargest(8, 0).tail(1)[0].values[0]

    dateTimeIndex = pd.date_range(start=date_starting, periods=52, freq="W")

    # 7 greatest performance stocks of y_pred
    ticker_list = (
        y[thisYearMask]
        .reset_index(drop=True)[bl_bestStocks & bl_safeStocks]["Ticker"]
        .values
    )
    ##### Change in code for Z score filtering #####

    # After we know our stock picks, we get the stock performance
    # Get DataFrame index of time stamp, series of stock prices,
    # keys=tickers
    stockRet = getStockTimeSeries(
        dateTimeIndex, y_withData, ticker_list, thisYearMask, daily_stock_prices
    )

    # Get DataFrame of relative stock prices from 1st day(or close)
    # and whole portfolio
    stockRetRel = getPortfolioRelativeTimeSeries(stockRet)
    return [stockRetRel, stockRetRel["Portfolio"], ticker_list]


def getPortTimeSeries(
    y_withData, X, daily_stock_prices, ml_model_pipeline, verbose=True
):
    """
    Returns DataFrames of selected stocks/portfolio performance since 2009.
    Needs X and y(with data), the daily_stock_prices DataFrame,
    the model pipeline we want to test.
    X is standard X for model input.
    y_withData is the stock price before/after df with date information.
    Input X and y must be data that the model was not trained on.
    """

    wrk_y_withData = pd.DataFrame()
    wrk_y_withData["Year"] = y_withData["Date"].dt.year
    years_in_y_withData = wrk_y_withData.groupby("Year")["Year"].count()

    first_year = years_in_y_withData.index[0]
    last_year = years_in_y_withData.index[-1]

    first_date = str(first_year) + "-01-01"
    num_years = last_year - first_year

    # set date range to make stock picks over
    dr = pd.date_range(
        start=first_date, periods=num_years, freq="YE"
    ) + pd.to_timedelta(
        "9w"
    )  # start every March
    # For each date in the date_range, make stock selections
    # and plot the return results of those stock selections
    port_perf_all_years = pd.DataFrame()
    perfRef = 1  # performance starts at 1.

    for curr_date in dr:

        # Get performance for this year
        [comp, this_year_perf, ticker_list] = getPortTimeSeriesForYear(
            curr_date, y_withData, X, daily_stock_prices, ml_model_pipeline
        )

        if verbose:  # If you want text output
            print(
                "Backtest performance for year starting ",
                curr_date,
                " is:",
                round((this_year_perf.iloc[-1] - 1) * 100, 2),
                "%",
            )
            print("With stocks:", ticker_list)
            for tick in ticker_list:
                print(
                    tick,
                    "Performance was:",
                    round((comp[tick].iloc[-1] - 1) * 100, 2),
                    "%",
                )
            print("---------------------------------------------")

        # Stitch performance for every year together
        this_year_perf = this_year_perf * perfRef
        # print(comp)
        # port_perf_all_years = pd.concat([port_perf_all_years, this_year_perf])

        port_perf_all_years = (
            port_perf_all_years.copy()
            if this_year_perf.empty
            else (
                pd.DataFrame(this_year_perf)
                if port_perf_all_years.empty
                else pd.concat([port_perf_all_years, this_year_perf])
            )
        )

        perfRef = this_year_perf.iloc[-1]

    # Return portfolio performance for all years
    port_perf_all_years.columns = ["Indexed Performance"]
    return port_perf_all_years


# Linear model pipeline
def trainLinearModel(X_train, y_train):
    pl_linear = Pipeline(
        [("Power Transformer", PowerTransformer()), ("linear", LinearRegression())]
    )
    pl_linear.fit(X_train, y_train)
    return pl_linear


# ElasticNet model pipeline
def trainElasticNetModel(X_train, y_train):
    pl_ElasticNet = Pipeline(
        [
            ("Power Transformer", PowerTransformer()),
            ("ElasticNet", ElasticNet(l1_ratio=0.00001)),
        ]
    )
    pl_ElasticNet.fit(X_train, y_train)
    return pl_ElasticNet


# KNeighbors regressor
def trainKNeighborsModel(X_train, y_train):
    pl_KNeighbors = Pipeline(
        [
            ("Power Transformer", PowerTransformer()),
            ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=40)),
        ]
    )
    pl_KNeighbors.fit(X_train, y_train)
    return pl_KNeighbors


# DecisionTreeRegressor
def traindecTreeModel(X_train, y_train):
    pl_decTree = Pipeline(
        [
            (
                "DecisionTreeRegressor",
                DecisionTreeRegressor(max_depth=20, random_state=42),
            )
        ]
    )
    pl_decTree.fit(X_train, y_train)
    return pl_decTree


# RandomForestRegressor
def trainrfregressorModel(X_train, y_train):
    pl_rfregressor = Pipeline(
        [
            (
                "RandomForestRegressor",
                RandomForestRegressor(max_depth=10, random_state=42),
            )
        ]
    )
    pl_rfregressor.fit(X_train, y_train)

    return pl_rfregressor


# GradientBoostingRegressor
def traingbregressorModel(X_train, y_train):
    pl_GradBregressor = Pipeline(
        [
            (
                "GradBoostRegressor",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=14,
                    random_state=42,
                    loss="absolute_error",
                ),
            )
        ]
    )
    pl_GradBregressor.fit(X_train, y_train)

    return pl_GradBregressor


# SVM
def trainsvmModel(X_train, y_train):
    pl_svm = Pipeline(
        [
            ("Power Transformer", PowerTransformer()),
            ("SVR", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
        ]
    )
    pl_svm.fit(X_train, y_train)
    return pl_svm


# XGBoost
def trainxgbregressorModel(X_train, y_train):
    pl_xgbregressor = Pipeline(
        [
            ("XGBoost", xgb.XGBRegressor()),
        ]
    )
    pl_xgbregressor.fit(X_train, y_train)

    return pl_xgbregressor


def fixNansInX(x):
    """
    Takes in x DataFrame, edits it so that important keys
    are 0 instead of NaN.
    """
    keyCheckNullList = [
        "Short Term Debt",
        "Long Term Debt",
        "Interest Expense, Net",
        "Income Tax (Expense) Benefit, Net",
        "Cash, Cash Equivalents & Short Term Investments",
        "Property, Plant & Equipment, Net",
        "Revenue",
        "Gross Profit",
        "Total Current Liabilities",
    ]
    x[keyCheckNullList] = x[keyCheckNullList].fillna(0)
    x["Property, Plant & Equipment, Net"] = x[
        "Property, Plant & Equipment, Net"
    ].fillna(0)


def addColsToX(x):
    """
    Takes in x DataFrame, edits it to include:
        Enterprise Value.
        Earnings before interest and tax.

    """
    x["EV"] = (
        x["Market Cap"]
        + x["Long Term Debt"]
        + x["Short Term Debt"]
        - x["Cash, Cash Equivalents & Short Term Investments"]
    )

    x["EBIT"] = (
        x["Net Income (Common)"]
        - x["Interest Expense, Net"]
        - x["Income Tax (Expense) Benefit, Net"]
    )

    x["FCF"] = (
        x["Net Cash from Operating Activities"]
        + x["Change in Fixed Assets & Intangibles"]
    )


# Make new X with ratios to learn from.
def getXRatios(x_):
    """
    Takes in x_, which is the fundamental stock DataFrame raw.
    Outputs X, which is the data encoded into stock ratios.
    """
    X = pd.DataFrame()

    # 1. EV/EBIT
    X["EV/EBIT"] = x_["EV"] / x_["EBIT"]

    # 2. Op. In./(NWC+FA)
    X["Op. In./(NWC+FA)"] = x_["Operating Income (Loss)"] / (
        x_["Total Current Assets"]
        - x_["Total Current Liabilities"]
        + x_["Property, Plant & Equipment, Net"]
    )

    # 3. P/E
    X["P/E"] = x_["Market Cap"] / x_["Net Income (Common)"]

    # 4. P/B
    X["P/B"] = x_["Market Cap"] / x_["Total Equity"]

    # 5. P/S
    X["P/S"] = x_["Market Cap"] / x_["Revenue"]

    # 6. Op. In./Interest Expense
    X["Op. In./Interest Expense"] = (
        x_["Operating Income (Loss)"] / -x_["Interest Expense, Net"]
    )

    # 7. Working Capital Ratio
    X["Working Capital Ratio"] = (
        x_["Total Current Assets"] / x_["Total Current Liabilities"]
    )

    # 8. Return on Equity
    X["RoE"] = x_["Net Income (Common)"] / x_["Total Equity"]

    # 9. Return on Capital Employed
    X["ROCE"] = x_["EBIT"] / (x_["Total Assets"] - x_["Total Current Liabilities"])

    # 10. Debt/Equity
    X["Debt/Equity"] = x_["Total Liabilities"] / x_["Total Equity"]

    # 11. Debt Ratio
    X["Debt Ratio"] = x_["Total Assets"] / x_["Total Liabilities"]

    # 12 . Cash Ratio
    X["Cash Ratio"] = (
        x_["Cash, Cash Equivalents & Short Term Investments"]
        / x_["Total Current Liabilities"]
    )

    # 13. Asset Turnover
    X["Asset Turnover"] = x_["Revenue"] / x_["Property, Plant & Equipment, Net"]

    # 14. Gross Profit Margin
    X["Gross Profit Margin"] = x_["Gross Profit"] / x_["Revenue"]

    ### Altman ratios ###
    # 15. (CA-CL)/TA
    X["(CA-CL)/TA"] = (
        x_["Total Current Assets"] - x_["Total Current Liabilities"]
    ) / x_["Total Assets"]

    # 16. RE/TA
    X["RE/TA"] = x_["Retained Earnings"] / x_["Total Assets"]

    # 17. EBIT/TA
    X["EBIT/TA"] = x_["EBIT"] / x_["Total Assets"]

    # 18. Book Equity/TL
    X["Book Equity/TL"] = x_["Total Equity"] / x_["Total Liabilities"]

    # 19. Dividends Yield
    # X["Dividends Yield"] = x_["Dividends Paid"] / x_["Market Cap"]

    # 20. P/FCF
    X["P/FCF"] = x_["Market Cap"] / x_["FCF"]

    # 21. Altman Z score
    X["Z score"] = (
        3.25
        + 6.51 * X["(CA-CL)/TA"]
        + 3.26 * X["RE/TA"]
        + 6.72 * X["EBIT/TA"]
        + 1.05 * X["Book Equity/TL"]
    )

    X.fillna(0, inplace=True)
    return X


def fixXRatios(X):
    """
    Takes in X, edits it to have the distributions clipped.
    The distribution clippings are done manually by eye,
    with human judgement based on the information.
    """
    df_X = X.mask(np.isinf)
    df_max = df_X.max()
    df_min = df_X.min()
    for col in X.columns:
        min = df_min[col]
        max = df_max[col]
        if min >= max:
            print("Error en fixRatios ", col, min, max)
        X[col] = X[col].clip(min, max)


def getYPerf(y_):
    """
    Takes in y_, which has the stock prices and their respective
    dates they were that price.
    Returns a DataFrame y containing the ticker and the
    relative change in price only.
    """
    y = pd.DataFrame()
    y["Ticker"] = y_["Ticker"]
    y["Perf"] = (y_["Open Price2"] - y_["Open Price"]) / y_["Open Price"]
    y["Date"] = y_["Date"]
    y["Open Price"] = y_["Open Price"]
    y["Date2"] = y_["Date2"]
    y["Open Price2"] = y_["Open Price2"]
    # y["Perf"].fillna(0, inplace=True)
    y["Perf"] = y["Perf"].fillna(0)
    return y


def generateDataFile(localPath):
    """
    Generate Data & Clean it
    """
    X = getXDataMerged(localPath)

    y = getY(localPath, X)
    y = pd.DataFrame(
        y,
        columns=[
            "Ticker",
            "Open Price",
            "Date",
            "Volume",
            "Ticker2",
            "Open Price2",
            "Date2",
            "Volume2",
        ],
    )

    data = pd.concat(
        [
            X,
            y[
                [
                    "Open Price",
                    "Date",
                    "Volume",
                    "Open Price2",
                    "Date2",
                    "Volume2",
                ]
            ],
        ],
        axis=1,
    )

    companies_count = data.groupby("Fiscal Year")["Fiscal Year"].count()

    MINIMUM_COMPANIES_COUNT = 500

    first_year_to_keep = companies_count.loc[
        companies_count > MINIMUM_COMPANIES_COUNT
    ].index[0]

    data = data[data["Fiscal Year"] >= first_year_to_keep]

    data.sort_values(["Fiscal Year", "Ticker"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Before data cleaning data.shape: ", data.shape)
    # ****************************************************************
    # * clear data (Begin)
    # ****************************************************************
    bool_list1 = ~data["Volume"].isnull()
    data = data[bool_list1]

    # Issue where no share price
    bool_list2 = ~data["Open Price"].isnull()

    # Issue where there is low/no volume
    bool_list3 = ~(data["Volume"] < 1e4)
    data = data[bool_list2 & bool_list3]

    # Issues where no listed number of shares
    bool_list4 = ~data["Shares (Diluted)_x"].isnull()
    data = data[bool_list4]

    # Issues where no ticker
    bool_list5 = ~data["Ticker"].isnull()
    data = data[bool_list5]

    # Issue where dates missing(Removes latest data too, which we can't use)
    bool_list6 = ~data["Date2"].isnull()
    # bool_list7 = ~data["Volume2"].isnull()

    # Issue where there is low/no volume
    # bool_list8 = (data["Volume2"] < 1e4) # Hay Volume2 que son invalid value

    data_train = data[bool_list6]

    y = pd.DataFrame()
    y["Perf"] = (data_train["Open Price2"] - data_train["Open Price"]) / data_train[
        "Open Price"
    ]
    data["Perf"] = y["Perf"]

    # ****************************************************************
    # * clear data (End)
    # ****************************************************************

    data["Market Cap"] = data["Open Price"] * data["Shares (Diluted)_x"]

    print("After data cleaning data.shape: ", data.shape)
    data.to_csv("Annual_Data_Complete.csv")


def getData(localPath):
    data_file_name = "Annual_Data_Complete.csv"

    if not os.path.isfile(data_file_name):
        print("Generating Data File")
        generateDataFile(localPath)

    data = pd.read_csv(data_file_name, parse_dates=["Date", "Date2"], index_col=0)
    return data


def getX(data):
    fixNansInX(data)
    addColsToX(data)
    X = getXRatios(data)
    fixXRatios(X)

    for c in X.columns:
        X[c] = X[c].astype(float)

    return X


def getY_(data):
    y = pd.DataFrame()
    for c in ["Ticker", "Perf", "Date", "Open Price", "Date2", "Open Price2"]:
        if c in ["Perf", "Open Price", "Open Price2"]:
            y[c] = data[c].astype(float)
        else:
            y[c] = data[c]

    return y


def getXy(localPath):
    """
    Feature Engineering
    https://github.com/Damonlee92/Build_Your_Own_AI_Investor_2024/blob/main/Chapter_5_to_7_AI_and_Backtesting/2_Process_X_Y_Learning_Data.ipynb
    """
    data = getData(localPath)

    bool_list = ~data["Perf"].isnull()

    data = data[bool_list]

    X = getX(data)
    y = getY_(data)

    y = pd.DataFrame()
    y = data[["Ticker", "Perf", "Date", "Open Price", "Date2", "Open Price2"]]

    print("X.shape : ", X.shape)
    print("y.shape : ", y.shape)

    return X, y


def getTrainedPipeline(model: str, X_train, y_train):
    if model == "LinearRegression":
        model_pl = trainLinearModel(X_train, y_train)
    if model == "ElasticNet":
        model_pl = trainElasticNetModel(X_train, y_train)
    if model == "KNeighborsRegressor":
        model_pl = trainKNeighborsModel(X_train, y_train)
    if model == "RandomForestRegressor":
        model_pl = trainrfregressorModel(X_train, y_train)
    if model == "DecisionTreeRegressor":
        model_pl = traindecTreeModel(X_train, y_train)
    if model == "GradientBoostingRegressor":
        model_pl = traingbregressorModel(X_train, y_train)
    if model == "SVR":
        model_pl = trainsvmModel(X_train, y_train)
    if model == "XGBRegressor":
        model_pl = trainxgbregressorModel(X_train, y_train)

    return model_pl


def calculateSPYReturns(start_date, end_date):
    spy = df = yf.download(
        "SPY",
        start=start_date,
        end=end_date,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )
    spy["Relative"] = spy["Open"] / spy["Open"].iloc[0]
    spy_perf = spy["Relative"].iloc[-1] - spy["Relative"].iloc[0]

    spy_perf = spy["Relative"].iloc[-1]
    spy_vol = spy["Relative"].diff().std() * np.sqrt(52)
    return spy_perf, spy_vol


spy_results_generated = False


def getResultsForModel(
    X,
    y_pec,
    y_withData,
    daily_stock_prices,
    model_pipeline_list,
    runs_per_model=1,
    thead_number=0,
    verbose=True,
):
    """
    getResultsForModel
    Choose the model pipelines to run loop for.
    """
    global spy_results_generated
    i, results = 0, []
    for model in model_pipeline_list:

        if verbose:
            iter = range(0, runs_per_model)
        else:
            bar_desc = "{:>25}".format(model)
            iter = tqdm(
                range(0, runs_per_model),
                desc=bar_desc,
                position=thead_number + 1,
                leave=False,
            )

        for test_num in iter:
            X_train, X_test, y_train, y_test = train_test_split(X, y_pec, test_size=0.5)
            # Train different models
            model_pl = getTrainedPipeline(model, X_train, y_train)
            y_withData_Test = y_withData.loc[X_test.index]
            # Here is our backtesting code
            test = getPortTimeSeries(
                y_withData_Test, X_test, daily_stock_prices, model_pl, verbose=False
            )
            perf = test["Indexed Performance"].iloc[-1]
            vol = test["Indexed Performance"].diff().std() * np.sqrt(52)
            if verbose:
                print("Performed test ", i, [i, model, perf, vol])
            results.append([i, model, perf, vol])
            if not spy_results_generated:
                spy_perf, spy_vol = calculateSPYReturns(
                    test.index.min(), test.index.max()
                )
                results.append([i, "SPY", spy_perf, spy_vol])
                spy_results_generated = True
            i = i + 1

    # Save our results for plotting
    results_df = pd.DataFrame(
        results,
        columns=[
            "Test Number",
            "Model Used",
            "Indexed Return",
            "Annual Volatility",
        ],
    )
    # Append to an existing results file if available,
    # else make new results file.
    # In parallel there is an extremely remote chance
    # two cores try and access file at same time.
    # To keep code simple this is OK.
    bt_statistics_fname = "Backtest_statistics.csv"
    if os.path.isfile(bt_statistics_fname):
        results_df.to_csv(bt_statistics_fname, mode="a", header=False)
    else:
        results_df.to_csv(bt_statistics_fname)


def plotBacktestDist(results_df, model_file, col):
    data = results_df[results_df["Model Used"] == model_file][col]
    ax = data.hist(bins=50, density=True, alpha=0.7)
    ax2 = data.plot.kde(alpha=0.9)
    max_val = data.max()
    ax.set_xlabel(col)
    ax.set_ylabel("Normalised Frequency")
    ax.set_title(
        "{} Backtest Distribution for {}, {} Runs".format(
            col,
            model_file,
            data.size,
        ),
        fontsize=10,
    )
    ax.grid()
    mean = data.mean()
    ymin, ymax = ax.get_ylim()
    if col == "Indexed Return":
        # Plot S&P 500 returns
        spy_return = results_df[results_df["Model Used"] == "SPY"][col].iloc[0]

        ax.plot(
            [spy_return, spy_return],
            [ymin, ymax],
            color="r",
            linestyle="-",
            linewidth=1.5,
            alpha=1,
        )
        ax.plot(
            [mean, mean],
            [ymin, ymax],
            color="lime",
            linestyle="--",
            linewidth=1.5,
            alpha=1,
        )
        # plt.xlim(0, 15)
    if col == "Annual Volatility":
        # Plot S&P 500 volatility
        spy_vol = results_df[results_df["Model Used"] == "SPY"][col].iloc[0]
        ax.plot(
            [spy_vol, spy_vol],
            [ymin, ymax],
            color="r",
            linestyle="-",
            linewidth=2,
        )
        ax.plot([mean, mean], [ymin, ymax], color="lime", linestyle="--", linewidth=2)
        # plt.xlim(0, 1.5)
    ax.legend(
        [
            "Fitted Smooth Kernel",
            "S&P500 Benchmark",
            "Simulation Mean {}".format(round(mean, 2)),
            "Simulation Backtests",
        ],
        loc="upper right",
    )
