#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# * ---------------------------------------------------------------- *
# *
# * Project Description:
# *
# * ---DATE--   --DESCRIPTION------------------------------ -AUTHOR- *
# * 03Sep2025   Creation                                     JFRD
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

import matplotlib as mpl
from matplotlib import pyplot as plt

from dotenv import load_dotenv

import pandas as pd
import numpy as np

import yfinance as yf

import threading

import quantstats_lumi as qs

from AI_Invest_Assistant.utils import getStatement
from AI_Invest_Assistant.utils import getXDataMerged
from AI_Invest_Assistant.utils import getYRawData
from AI_Invest_Assistant.utils import getData
from AI_Invest_Assistant.utils import getX
from AI_Invest_Assistant.utils import getY
from AI_Invest_Assistant.utils import getXy
from AI_Invest_Assistant.utils import getStockPricesIndexed
from AI_Invest_Assistant.utils import getTrainedPipeline
from AI_Invest_Assistant.utils import getPortTimeSeries
from AI_Invest_Assistant.utils import getRegressorList
from AI_Invest_Assistant.utils import getResultsForModel
from AI_Invest_Assistant.utils import plotBacktestDist
from AI_Invest_Assistant.utils import calcZScores
from AI_Invest_Assistant.utils import generateDataFile
from AI_Invest_Assistant.utils import getTickerPerformance

# from AI_Invest_Assistant.utils import getStockPriceBetweenDates
# from AI_Invest_Assistant.utils import getStockTimeSeries
# from AI_Invest_Assistant.utils import getPortfolioRelativeTimeSeries

# * --------------------- *
# *    Globals (Begin)    *
# * --------------------- *

DATA_DIR = "data"
OUTPUT_DATA_DIR = DATA_DIR + "\\" + "output" + "\\"
INPUT_DATA_DIR = DATA_DIR + "\\" + "input" + "\\"
CACHE_DATA_DIR = DATA_DIR + "\\" + "cache" + "\\"

# * --------------------- *
# *    Globals (End)      *
# * --------------------- *


def handler(signum, frame):
    print("Signal handler called with signal", signum)
    sys.exit(-1)


def getBackTestStatsReportEntry(model_name, returns):
    total_return = round(qs.stats.comp(returns) * 100, 2)
    total_return = f"{total_return:3.2f} %"
    volatility = round(qs.stats.volatility(returns * 100, periods=52), 2)
    volatility = f"{volatility:3.2f} %"
    sortino = round(qs.stats.sortino(returns, periods=52), 2)
    sharpe = round(qs.stats.sharpe(returns, periods=52), 2)
    return [model_name, total_return, volatility, sharpe, sortino]


def generateBackTestStatsReport(train, test, stats_report):
    stats_report_df = pd.DataFrame(
        stats_report,
        columns=["Model", "Total Return", "Volatility", "Sharpe", "Sortino"],
    )

    stats_report_df.sort_values(["Sortino", "Sharpe"], inplace=True, ascending=False)
    stats_report_df.reset_index(drop=True, inplace=True)

    print("")
    print(f"Train Period from {train[0]} to {train[-1]}")
    print(f"Test Period from {test[0]} to {test[-1]}")
    print("")
    print(stats_report_df)
    print("")

    getBackTestEOYReportEntry


def getBackTestEOYReportEntry(model_name, returns):
    eoy = qs.stats.monthly_returns(returns)
    eoy.reset_index(inplace=True)

    eoy_df = pd.DataFrame()
    eoy_df["Date"] = eoy["index"]

    eoy_df["Return"] = round(eoy["EOY"] * 100, 2)
    eoy_df["Return"] = eoy_df["Return"].astype(str)
    eoy_df["Return"] = eoy_df["Return"] + " %"

    eoy_df.set_index("Date", inplace=True)

    t_eoy_df = eoy_df.transpose()

    eoy_dict = t_eoy_df.iloc[0].to_dict()

    eoy_list = list(eoy_dict.items())
    eoy_list.insert(0, ("Model", model_name))

    eoy_dict = dict(eoy_list)

    return eoy_dict


def generateBackTestEOYReport(train, test, eoy_report, keys):
    eoy_report_df = pd.DataFrame(
        eoy_report,
        columns=keys,
    )
    print("")
    print(f"EOY (Calendar Years) Report")
    print("")
    print(eoy_report_df)
    print("")


def generateHTMLReport(returns, model_name, benchmark):
    output_name = OUTPUT_DATA_DIR + "\\" + model_name + "_" + benchmark + ".html"
    output_name = os.path.normpath(output_name)
    qs.reports.html(
        returns,
        title="Strategy: " + model_name,
        output=output_name,
        periods_per_year=52,
        benchmark=benchmark,
        benchmark_period=52,
    )
    pass


def backTest(localPath, verbose=True):
    X, y = getXy(localPath)

    daily_stock_prices_data = getStockPricesIndexed(localPath)

    train_mask = y["Date"].dt.year < 2016

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[~train_mask]
    y_test = y.loc[~train_mask]

    yperf = y_train["Perf"]

    models = [
        "LinearRegression",
        "KNeighborsRegressor",
        "GradientBoostingRegressor",
        "RandomForestRegressor",
        "XGBRegressor",
    ]

    spy_RetRel = None
    stats_report = []
    EOY_report = []
    EOY_keys = []

    for model_name in models:
        # Train model
        trained_model_pipeline = getTrainedPipeline(model_name, X_train, yperf)

        if verbose:
            print(f"***************************************************")
            print(f"* Regressor {model_name}")
            print(f"***************************************************")

        backTest = getPortTimeSeries(
            y_test,
            X_test,
            daily_stock_prices_data,
            trained_model_pipeline,
            verbose=True,
        )

        backTest["returns"] = backTest["Indexed Performance"].pct_change()
        backTest.dropna(inplace=True)

        if spy_RetRel is None:
            start_date = backTest.index[0]
            end_date = backTest.index[-1]

            spy_RetRel = getTickerPerformance(
                "SPY", y_test, start_date, end_date, daily_stock_prices_data
            )

            spy_RetRel["returns"] = spy_RetRel["Portfolio"].pct_change()
            spy_RetRel.dropna(inplace=True)

            stats_report.append(
                getBackTestStatsReportEntry("SPY", spy_RetRel["returns"])
            )

            spy_eof_dict = getBackTestEOYReportEntry("SPY", spy_RetRel["returns"])
            EOY_keys = list(spy_eof_dict.keys())
            EOY_report.append(list(spy_eof_dict.values()))

        # remove first row
        backTest = backTest.iloc[1:]

        stats_report.append(
            getBackTestStatsReportEntry(model_name, backTest["returns"])
        )

        EOY_report.append(
            list(getBackTestEOYReportEntry(model_name, backTest["returns"]).values())
        )

        generateHTMLReport(backTest["returns"], model_name, "SPY")

    train = y_train["Date"].dt.date.values

    test = [x.date() for x in spy_RetRel.index]

    generateBackTestStatsReport(train, test, stats_report)
    generateBackTestEOYReport(train, test, EOY_report, EOY_keys)


def generateTestDataThreading(
    localPath, model_pipeline_list, X, y, yperf, daily_stock_prices
):
    print("generateTestDataThreading")

    l = len(model_pipeline_list)
    thread_list = []

    for i in range(l):
        thread = threading.Thread(
            target=getResultsForModel,
            args=(
                X,
                yperf,
                y,
                daily_stock_prices,
                [model_pipeline_list[i]],
                30,
                i,
                False,
            ),
        )
        thread_list.append(thread)
        thread.daemon = True
        thread.start()
        # print(f"Thread {i} started.")

    # Wait for threads to finish
    while True:
        if not any([thread.is_alive() for thread in thread_list]):
            # All threads have stopped
            break
        else:
            # Some threads are still going
            time.sleep(1)


def generateTestDataSequential(
    localPath, model_pipeline_list, X, y, yperf, daily_stock_prices
):
    print("generateTestDataSequential")

    l = len(model_pipeline_list)
    thread_list = []

    for i in range(l):
        getResultsForModel(
            X, yperf, y, daily_stock_prices, [model_pipeline_list[i]], 30, i, False
        )


def generateTestData(localPath):
    model_pipeline_list = getRegressorList()

    X, y = getXy(localPath)
    yperf = y["Perf"]

    daily_stock_prices = getStockPricesIndexed(localPath)

    method = "Threading"
    if method == "Threading":
        generateTestDataThreading(
            localPath, model_pipeline_list, X, y, yperf, daily_stock_prices
        )
    elif method == "Sequential":
        generateTestDataSequential(
            localPath, model_pipeline_list, X, y, yperf, daily_stock_prices
        )


def plotTestData(localPath):
    bt_statistics_fname = "Backtest_statistics.csv"
    if not os.path.isfile(bt_statistics_fname):
        print("Generating Test Data")
        generateTestData(localPath)

    results_df = pd.read_csv(bt_statistics_fname, index_col=0)

    model_pipeline_list = getRegressorList()

    # model_file = "LinearRegression"
    for model_file in model_pipeline_list:
        # quantstats_lumi change the mathplotlib default values
        mpl.rcParams.update(mpl.rcParamsDefault)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plotBacktestDist(results_df, model_file, "Indexed Return")
        plt.subplot(1, 2, 2)
        plotBacktestDist(results_df, model_file, "Annual Volatility")
        plt.show()


def printResults(model_name, stockPicks):
    stockPicks["Today Price"] = np.nan
    stockPicks["% Change"] = np.nan

    for i in range(len(stockPicks)):
        row = stockPicks.iloc[i]

        try:
            stock = yf.Ticker(row["Ticker"])
            price = stock.info["regularMarketPrice"]
        except Exception as e:
            print("Fallo yfinance : ", e, row["Ticker"])
            price = np.nan

        stockPicks.loc[i, "Today Price"] = price

    stockPicks["% Change"] = (
        stockPicks["Today Price"] - stockPicks["Open Price"]
    ) / stockPicks["Open Price"]

    stockPicks.dropna(inplace=True)
    stockPicks.reset_index(drop=True, inplace=True)

    print("")
    print(f"***** {model_name}")
    print("")
    print(stockPicks.head(7))
    print("")
    print(
        f"mean Perf. Score {stockPicks.head(7)["Perf. Score"].mean():+.4f}",
    )
    print(
        f"   mean % change {stockPicks.head(7)["% Change"].mean():+.4f}",
    )


def stockPicks(localPath):
    X, y = getXy(localPath)
    yperf = y["Perf"].astype(float)

    # **********************************************
    # Get X data for prediction (Begin)
    # **********************************************
    data = getData(localPath)

    # Get the current year
    wrk_data = pd.DataFrame()
    wrk_data["Publish Year"] = pd.to_datetime(data["Publish Date"]).dt.year
    years_in_data = wrk_data.groupby("Publish Year")["Publish Year"].unique()
    this_year = years_in_data.index[-1]

    PublishDateStart = str(this_year) + "-01-01"
    PublishDateEnd = str(this_year) + "-03-01"

    bool_list = pd.to_datetime(data["Publish Date"]).between(
        pd.to_datetime(PublishDateStart), pd.to_datetime(PublishDateEnd)
    )
    x_ = data[bool_list].copy()
    x_.reset_index(drop=True, inplace=True)

    X_curr = getX(x_)
    # **********************************************
    # Get X data for prediction (End)
    # **********************************************

    models = [
        "LinearRegression",
        "KNeighborsRegressor",
        "GradientBoostingRegressor",
        "RandomForestRegressor",
        "XGBRegressor",
    ]

    for model_name in models:
        x_wrk_ = x_.copy()
        ml_model_pipeline = getTrainedPipeline(model_name, X, yperf)

        # Get return prediction from model
        y_pred = ml_model_pipeline.predict(X_curr)

        # Make it a DataFrame to select the top picks
        y_pred = pd.DataFrame(y_pred)

        ##### Change in code for Z score filtering #####
        # Separate out stocks with low Z scores
        z = calcZScores(X_curr)

        # 3.75 is approx. B- rating
        bl_safeStocks = z["Z score"] > 2

        x_wrk_["Perf. Score"] = y_pred
        x_wrk_ = x_wrk_[bl_safeStocks]
        x_wrk_.reset_index(drop=True, inplace=True)
        x_wrk_.sort_values("Perf. Score", inplace=True, ascending=False)

        stockPicks = x_wrk_.head(20)[
            ["Ticker", "Publish Date", "Perf. Score", "Open Price"]
        ]
        stockPicks.reset_index(drop=True, inplace=True)

        printResults(model_name, stockPicks)


def featureImportance(localPath):
    # https://builtin.com/data-science/dimensionality-reduction-python
    X, y = getXy(localPath)
    yperf = y["Perf"].astype(float)

    ml_model_pipeline = getTrainedPipeline("RandomForestRegressor", X, yperf)
    model = ml_model_pipeline["RandomForestRegressor"]

    feature_df = pd.DataFrame(
        {"Importance": model.feature_importances_, "Features": X.columns}
    )

    feature_df = feature_df.sort_values("Importance", ascending=False)
    print(feature_df.head(20)[["Features", "Importance"]].to_string(index=False))

    plt.bar(feature_df["Features"], feature_df["Importance"])
    plt.xticks(rotation=90)
    plt.title("Random Forest Model Feature Importance")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--generate-data-file",
        help="Generate Data & Clean it",
        action="store_true",
    )

    parser.add_argument(
        "-b",
        "--back-test",
        help="backtest",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--generate-test-data",
        help="Generate test data file",
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--plot-test-data",
        help="Generate test data file",
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--stock-picks",
        help="Print stock picks",
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--feature-importance",
        help="Plot feature importance",
        action="store_true",
    )

    args = parser.parse_args()

    if args.generate_data_file:
        generateDataFile(localPath=INPUT_DATA_DIR)

    if args.back_test:
        backTest(localPath=INPUT_DATA_DIR)

    if args.generate_test_data:
        generateTestData(localPath=INPUT_DATA_DIR)

    if args.plot_test_data:
        plotTestData(localPath=INPUT_DATA_DIR)

    if args.stock_picks:
        stockPicks(localPath=INPUT_DATA_DIR)

    if args.feature_importance:
        featureImportance(localPath=INPUT_DATA_DIR)

    return 0


# * --------------------------- *
# *    Main Program (Begin)     *
# * --------------------------- *

if __name__ == "__main__":
    start = time.monotonic()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)

    if not os.path.exists(INPUT_DATA_DIR):
        os.makedirs(INPUT_DATA_DIR)

    if not os.path.exists(CACHE_DATA_DIR):
        os.makedirs(CACHE_DATA_DIR)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
    )

    signal.signal(signal.SIGINT, handler)
    load_dotenv()
    rc = main()

    print("Execution Time " + str(time.monotonic() - start) + " seconds")
    print("RC = " + str(rc))
    sys.exit(rc)

# * --------------------------- *
# *    Main Program (End)       *
# * --------------------------- *
