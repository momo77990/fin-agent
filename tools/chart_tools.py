import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.tools import tool
@tool
def plot_stock_chart(ticker: str, prices_json: str):
    """
    输入股票代码和价格JSON数据，绘制折线图并保存到本地 'chart.png'。
    注意：prices_json 必须是标准 JSON 字符串。
    """
    # 这里你需要做一些 JSON 解析和 Matplotlib 绘图逻辑
    # 核心是：plt.plot(), plt.savefig("static/chart.png")
    # 返回： "图表已生成，路径为 static/chart.png"
    try:
        df = pd.read_json(prices_json)
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['close'], marker='o')
        plt.title(f"{ticker} 股价走势")
        plt.xlabel("日期")
        plt.ylabel("收盘价")
        plt.grid(True)
        plt.savefig("static/chart.png")
        return "图表已生成，路径为 static/chart.png"
    except Exception as e:
        return f"生成图表时发生错误: {str(e)}"
