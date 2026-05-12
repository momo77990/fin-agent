"""一次性脚本:生成一份虚构的金融测试 PDF,用于验证 RAG 检索。

放到 data/ 目录(已通过 docker-compose 挂载到容器内 /app/data),
下次 Agent 调 search_filings 时会自动索引。
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib import colors


pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def build_styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1_cn", parent=base["Heading1"], fontName="STSong-Light",
        fontSize=18, leading=24, spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "H2_cn", parent=base["Heading2"], fontName="STSong-Light",
        fontSize=14, leading=20, spaceBefore=12, spaceAfter=8,
    )
    body = ParagraphStyle(
        "Body_cn", parent=base["BodyText"], fontName="STSong-Light",
        fontSize=11, leading=18, spaceAfter=6,
    )
    return h1, h2, body


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_abc_tech_report.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    )
    h1, h2, body = build_styles()
    story = []

    story.append(Paragraph("ABC 科技股份有限公司 2025 年度研究报告", h1))
    story.append(Paragraph("(测试用虚构数据 · 仅供 Fin-Agent RAG 检索验证)", body))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("发布机构: Fin-Agent 内部测试报告", body))
    story.append(Paragraph("报告日期: 2026 年 3 月 14 日", body))
    story.append(Paragraph("股票代码: ABC.NX (虚构)", body))
    story.append(PageBreak())

    story.append(Paragraph("第一章 公司概况", h1))
    story.append(Paragraph(
        "ABC 科技股份有限公司(以下简称『ABC 科技』)成立于 2011 年,"
        "总部位于上海张江,是一家专注于工业级边缘计算与机器视觉解决方案的高新技术企业。"
        "截至 2025 年末,公司员工总数为 4,287 人,其中研发人员占比 62.3%。"
        "公司于 2019 年在上海证券交易所科创板挂牌上市,虚构股票代码 ABC.NX。",
        body,
    ))
    story.append(Paragraph(
        "ABC 科技的核心业务分为三大板块:工业边缘计算盒子(EdgeBox 系列)、"
        "机器视觉一体机(VisionCube 系列)、以及面向汽车前装的 ADAS 视觉算法授权。"
        "2025 财年三大板块营收占比分别为 47.2%、31.8% 和 21.0%。",
        body,
    ))

    story.append(Paragraph("第二章 2025 财年核心财务数据", h2))
    fin_data = [
        ["指标", "2024 年", "2025 年", "同比变化"],
        ["营业收入(亿元)", "38.42", "52.71", "+37.2%"],
        ["归母净利润(亿元)", "4.18", "6.93", "+65.8%"],
        ["毛利率", "41.5%", "44.8%", "+3.3pp"],
        ["研发投入(亿元)", "5.92", "8.14", "+37.5%"],
        ["研发投入占营收比", "15.4%", "15.4%", "持平"],
        ["每股收益(元)", "0.87", "1.44", "+65.5%"],
    ]
    tbl = Table(fin_data, hAlign="LEFT", colWidths=[5.5*cm, 3*cm, 3*cm, 3*cm])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "STSong-Light", 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "2025 年公司营收同比增长 37.2%,净利润增速 65.8%,显著高于行业平均。"
        "毛利率提升主要得益于 EdgeBox Pro 系列在汽车制造客户的批量出货。",
        body,
    ))
    story.append(PageBreak())

    story.append(Paragraph("第三章 业务结构与重要客户", h1))
    story.append(Paragraph(
        "ABC 科技 2025 年前五大客户合计贡献营收 58.4%,集中度较 2024 年的 51.2% 进一步上升。"
        "其中第一大客户为某全球领先的汽车制造商,贡献营收占比 22.8%。",
        body,
    ))
    story.append(Paragraph(
        "EdgeBox 系列在 2025 年实现出货量 12.7 万台,同比增长 41.3%。"
        "公司预计 2026 年 EdgeBox 出货量将达到 18 万台左右,但毛利率可能从 44.8% 回落至 42% 附近,"
        "原因是新一代国产 7nm SoC 在量产初期成本仍高于上一代成熟工艺。",
        body,
    ))

    story.append(Paragraph("第四章 风险因素", h1))
    story.append(Paragraph("4.1 客户集中度风险", h2))
    story.append(Paragraph(
        "公司前五大客户营收占比高达 58.4%,若主要客户调整采购策略或自研替代,"
        "将对公司收入造成显著影响。管理层已启动『2026 多元化客户拓展计划』,"
        "目标是在 2026 年末把前五大客户占比降至 50% 以下。",
        body,
    ))
    story.append(Paragraph("4.2 供应链风险", h2))
    story.append(Paragraph(
        "EdgeBox Pro 所用的高端国产 SoC 目前仅有 1 家国内晶圆代工厂可量产,"
        "若该代工厂产能紧张或地缘政治升级,将影响公司 EdgeBox 系列的交付节奏。"
        "公司正与第二家代工厂洽谈备份产能,预计 2026 年下半年完成产线导入。",
        body,
    ))
    story.append(Paragraph("4.3 技术替代风险", h2))
    story.append(Paragraph(
        "机器视觉领域算法迭代迅速,Transformer-based 视觉大模型可能在 2-3 年内"
        "改变现有的卷积神经网络主导格局。公司在 2025 年研发投入中已划拨 1.8 亿元"
        "用于视觉大模型自研,但仍存在不及预期的可能。",
        body,
    ))

    story.append(Paragraph("第五章 管理层讨论与分析", h1))
    story.append(Paragraph(
        "首席执行官张明远先生在年报致辞中指出,"
        "2025 年是 ABC 科技从『盒子供应商』向『算力 + 算法平台』转型的关键一年。"
        "公司计划在 2026 年发布 VisionCube X1 一体机,该产品将整合自研视觉大模型,"
        "目标售价区间 18,000 至 28,000 元,首年出货目标 3 万台。",
        body,
    ))
    story.append(Paragraph(
        "首席财务官李静女士表示,公司 2025 年末账面现金及等价物 21.4 亿元,"
        "资产负债率 28.6%,处于行业较低水平。董事会已批准 2025 年度现金分红方案:"
        "每 10 股派发现金红利 4.50 元(含税),分红总额约 2.16 亿元。",
        body,
    ))

    doc.build(story)
    print(f"PDF 已生成: {out_path}")
    print(f"文件大小: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
