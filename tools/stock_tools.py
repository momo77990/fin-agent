import yfinance as yf
from langchain.tools import tool


@tool(response_format="content_and_artifact")
def get_stock_price(ticker: str):
    """
    输入美股或 A 股代码（如 NVDA、AAPL、600519.SS），获取最近 1 个月历史收盘价。
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty or "Close" not in hist.columns:
            return (
                f"未获取到 {ticker} 最近一个月股价数据。"
                "请确认股票代码是否正确，或稍后重试 Yahoo Finance 数据源。",
                None,
            )

        prices = hist[["Close"]].dropna().reset_index()
        if prices.empty:
            return (
                f"未获取到 {ticker} 最近一个月有效收盘价数据。"
                "请确认股票代码是否正确，或稍后重试 Yahoo Finance 数据源。",
                None,
            )

        prices["Date"] = prices["Date"].dt.strftime("%Y-%m-%d")
        closes = prices["Close"]
        start_close = float(closes.iloc[0])
        latest_close = float(closes.iloc[-1])
        change_pct = ((latest_close - start_close) / start_close) * 100
        high_close = float(closes.max())
        low_close = float(closes.min())
        start_date = prices["Date"].iloc[0]
        end_date = prices["Date"].iloc[-1]
        records = [
            {"date": row["Date"], "close": round(float(row["Close"]), 2)}
            for _, row in prices.iterrows()
        ]

        summary = (
            f"{ticker} 最近一个月股价摘要：\n"
            f"- 数据区间：{start_date} 至 {end_date}\n"
            f"- 最新收盘价：{latest_close:.2f}\n"
            f"- 区间涨跌幅：{change_pct:+.2f}%\n"
            f"- 区间最高收盘价：{high_close:.2f}\n"
            f"- 区间最低收盘价：{low_close:.2f}\n"
            "- 最近 5 个交易日：\n"
        )
        recent_lines = "\n".join(
            f"  - {item['date']}: {item['close']:.2f}"
            for item in records[-5:]
        )
        content = f"{summary}{recent_lines}"
        artifact = {
            "ticker": ticker,
            "summary": content,
            "prices": records,
            "stats": {
                "latest_close": round(latest_close, 2),
                "change_pct": round(change_pct, 2),
                "high_close": round(high_close, 2),
                "low_close": round(low_close, 2),
                "start_date": start_date,
                "end_date": end_date,
            },
        }
        return content, artifact
    except Exception as e:
        return f"获取 {ticker} 股价数据时发生错误: {str(e)}", None
