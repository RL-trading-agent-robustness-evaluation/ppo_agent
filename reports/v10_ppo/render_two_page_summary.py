from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/v10_ppo"
PAGE1 = OUT_DIR / "ppo_progress_summary_page1.png"
PAGE2 = OUT_DIR / "ppo_progress_summary_page2.png"
W, H = 1600, 900

BG = "#ffffff"
TEXT = "#202428"
MUTED = "#6f777d"
GREEN = "#a8ce95"
GREEN_DARK = "#4d7d3e"
GREEN_LIGHT = "#eaf4e4"
GREEN_PALE = "#f4f9f1"
PANEL = "#d9dee1"
ORANGE = "#d98a28"
ORANGE_LIGHT = "#fff2de"
BLUE = "#6387a5"
FONT_REG = r"C:\Windows\Fonts\msjh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msjhbd.ttc"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


class Canvas:
    def __init__(self):
        self.image = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.image)

    def text(self, x, y, value, size, *, bold=False, fill=TEXT, anchor="la"):
        self.d.text((x, y), value, font=font(size, bold), fill=fill, anchor=anchor)

    def cell_text(self, x, y, w, h, value, size=17, *, bold=False, fill=TEXT,
                  pad=12, align="left", max_lines=2):
        f = font(size, bold)
        lines, current = [], ""
        for character in value:
            candidate = current + character
            if self.d.textlength(candidate, font=f) <= w - 2 * pad:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = character
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            while self.d.textlength(lines[-1] + "…", font=f) > w - 2 * pad:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "…"
        line_h = size + 6
        yy = y + (h - len(lines) * line_h) / 2
        for line in lines:
            if align == "center":
                xx, anchor = x + w / 2, "ma"
            elif align == "right":
                xx, anchor = x + w - pad, "ra"
            else:
                xx, anchor = x + pad, "la"
            self.d.text((xx, yy), line, font=f, fill=fill, anchor=anchor)
            yy += line_h

    def table(self, x, y, widths, heights, rows, *, highlights=None,
              warnings=None, aligns=None, body_size=16, header_size=17):
        highlights, warnings = highlights or set(), warnings or set()
        aligns = aligns or ["left"] * len(widths)
        yy = y
        for r, (row, rh) in enumerate(zip(rows, heights)):
            fill = GREEN_LIGHT if r == 0 else (ORANGE_LIGHT if r in warnings else
                                               GREEN_PALE if r in highlights else BG)
            xx = x
            for c, (value, cw) in enumerate(zip(row, widths)):
                self.d.rectangle((xx, yy, xx + cw, yy + rh), fill=fill,
                                 outline=GREEN, width=1)
                self.cell_text(xx, yy, cw, rh, str(value),
                               header_size if r == 0 else body_size,
                               bold=(r == 0) or (r in highlights and c == 0),
                               align=aligns[c], max_lines=2)
                xx += cw
            yy += rh

    def header(self, page, subtitle, *, badge, badge_color=GREEN_DARK):
        self.text(66, 54, "PPO", 46, bold=True, anchor="lm")
        self.text(166, 55, subtitle, 20, fill=MUTED, anchor="lm")
        self.text(1534, 49, f"PAGE {page} / 2", 15, bold=True,
                  fill=badge_color, anchor="rm")
        self.d.rounded_rectangle((1325, 68, 1534, 100), radius=16,
                                 fill=GREEN_LIGHT if badge_color == GREEN_DARK else ORANGE_LIGHT,
                                 outline=badge_color, width=1)
        self.text(1429, 84, badge, 14, bold=True, fill=badge_color, anchor="mm")

    def footer(self, left, right):
        self.d.line((66, 850, 1534, 850), fill="#eceff0", width=1)
        self.text(66, 875, left, 13, fill="#8b9297", anchor="lm")
        self.text(1534, 875, right, 13, bold=True, fill=ORANGE, anchor="rm")


def pct(v):
    return f"{v:.1%}"


common = json.loads((ROOT / "reports/v10_ppo/comparison_2019_2021/COMPARISON.json")
                    .read_text(encoding="utf-8"))
known = json.loads((ROOT / "reports/v10_ppo/comparison_2022_2025/COMPARISON.json")
                   .read_text(encoding="utf-8"))
pilot = json.loads((ROOT / "reports/v10_ppo/pilot200k_v1/PILOT_SUMMARY.json")
                   .read_text(encoding="utf-8"))


# ------------------------------------------------------------------- page 1
c = Canvas()
c.header(1, "Training progress & same-window validation", badge="CURRENT SNAPSHOT")

spec_rows = [
    ["版本", "核心規格", "目前觀察"],
    ["V4", "10D；2010–18 train；NAV reward；200k；V4 daily execution",
     "成熟基準；2019–21 Sharpe 1.246 高於 SPY，但財富 1.900 略低於 1.980"],
    ["V9 Control", "20D；fold 3 train 至 2018；SMA200 gate＋5日 min-hold；NAV；500k",
     "換手顯著降低；2019–21 財富 1.812、Sharpe 1.008，未超越 V4／SPY"],
    ["V9 Relative", "同 V9 環境；training reward = log NAV - log SPY；500k",
     "2019–21 模型中最佳財富 1.940；Sharpe 1.163；仍略低於 SPY 財富"],
    ["V10 Streak", "沿用 V9；NAV＋正報酬連續 streak bonus；β=0.10、cap=5；200k pilot",
     "fold-1 paired pilot 對 NAV control 為 0/5 改善；暫不支持 reward improvement"],
]
c.table(66, 112, [175, 610, 749], [42, 57, 57, 57, 57], spec_rows,
        highlights={3}, warnings={4}, body_size=16, header_size=18)

c.text(66, 426, "同窗比較（2019–2021）", 24, bold=True, anchor="lm")
c.text(355, 426, "五 seeds 中位數；true NAV、10 bps；描述性比較", 15,
       fill=MUTED, anchor="lm")

metrics = common["median_metrics"]
bh = common["spy_buy_and_hold"]
comparison_rows = [["模型", "Final wealth", "CAGR", "Sharpe", "Volatility",
                    "Max DD", "Exposure", "Turnover", "Cost"]]
for label, key in (("V4", "v4"), ("V9 Control", "v9_control"),
                   ("V9 Relative", "v9_relative"), ("V10 Streak", "v10_streak")):
    m = metrics[key]
    comparison_rows.append([label, f"{m['final_wealth']:.3f}×", pct(m["cagr"]),
                            f"{m['sharpe_excess_cash']:.3f}", pct(m["annualized_volatility"]),
                            pct(m["max_drawdown"]), pct(m["average_exposure"]),
                            f"{m['total_turnover']:.1f}×", f"${m['total_transaction_cost']/1000:.1f}k"])
comparison_rows.append(["SPY B&H", f"{bh['final_wealth']:.3f}×", pct(bh["cagr"]),
                        f"{bh['sharpe_excess_cash']:.3f}", pct(bh["annualized_volatility"]),
                        pct(bh["max_drawdown"]), pct(bh["average_exposure"]),
                        f"{bh['total_turnover']:.1f}×", f"${bh['total_transaction_cost']/1000:.1f}k"])
c.table(66, 452, [190, 155, 125, 125, 150, 135, 140, 130, 130],
        [42, 48, 48, 48, 48, 48], comparison_rows, highlights={3},
        warnings={4}, aligns=["left"] + ["center"] * 8, body_size=15, header_size=15)

bullets = [
    "同窗 winner：SPY 的財富最高；V4 的 Sharpe 最高；V9 Relative 是四個 PPO 中財富最高。",
    f"V10 pilot 的 paired evidence：streak 對 NAV control 為 {pilot['positive_pairs']}/5 改善，median Δlog wealth {pilot['median_delta_log_wealth']:.3f}。",
    "V10 尚未完成正式 3 folds × 5 seeds；目前結果不能覆寫 frozen selection rule。",
]
yy = 758
for item in bullets:
    c.text(72, yy, "•", 23, bold=True, anchor="lm")
    c.text(98, yy, item, 17, anchor="lm")
    yy += 30
c.footer("PPO | 2019–2021 common window | five-seed medians | snapshot 2026-10-01 (Asia/Taipei)",
         "DESCRIPTIVE — NOT CAUSAL EVIDENCE")
c.image.save(PAGE1, format="PNG", optimize=True)


# ------------------------------------------------------------------- page 2
c = Canvas()
c.header(2, "Known-period diagnostic", badge="KNOWN PERIOD", badge_color=ORANGE)
c.text(66, 112, "2022–2025 延伸診斷", 27, bold=True, anchor="lm")
c.text(340, 112, "2022-01-03 → 2025-12-31｜1,002 steps｜已開封資料", 16,
       fill=ORANGE, anchor="lm")

dist = known["distribution"]
bh = known["spy_buy_and_hold"]
wins = known["wins_vs_buy_and_hold"]
known_rows = [["模型", "Final wealth", "CAGR", "Excess", "Sharpe", "Volatility",
               "Max DD", "Exposure", "Turnover", "勝 B&H"]]
for label, key in (("V4", "v4"), ("V9 Relative", "v9_relative_fold3"),
                   ("V10 Streak pilot", "v10_streak_pilot")):
    d = dist[key]
    known_rows.append([label, f"{d['final_wealth']['median']:.3f}×",
                       pct(d["cagr"]["median"]), pct(d["annualized_mean_excess_return"]["median"]),
                       f"{d['sharpe_excess_cash']['median']:.3f}",
                       pct(d["annualized_volatility"]["median"]),
                       pct(d["max_drawdown"]["median"]), pct(d["average_exposure"]["median"]),
                       f"{d['total_turnover']['median']:.1f}×", f"{wins[key]}/5"])
known_rows.append(["SPY B&H", f"{bh['final_wealth']:.3f}×", pct(bh["cagr"]),
                   pct(bh["annualized_mean_excess_return"]), f"{bh['sharpe_excess_cash']:.3f}",
                   pct(bh["annualized_volatility"]), pct(bh["max_drawdown"]),
                   pct(bh["average_exposure"]), f"{bh['total_turnover']:.1f}×", "—"])
c.table(66, 142, [170, 145, 110, 115, 110, 135, 125, 135, 120, 110],
        [42, 50, 50, 50, 50], known_rows, highlights={3},
        aligns=["left"] + ["center"] * 9, body_size=15, header_size=15)

# Wealth spread panel
panel_y = 425
c.d.rounded_rectangle((66, panel_y, 930, 715), radius=5, outline=PANEL, width=2)
c.text(86, panel_y + 28, "Final wealth seed spread", 22, bold=True, anchor="lm")
c.text(365, panel_y + 28, "線段 = min–max；圓點 = median；SPY = 固定基準", 14,
       fill=MUTED, anchor="lm")
chart_x0, chart_x1 = 275, 880
vmin, vmax = 1.0, 1.75

def sx(value):
    return chart_x0 + (value - vmin) / (vmax - vmin) * (chart_x1 - chart_x0)

for tick in (1.0, 1.2, 1.4, 1.6, 1.75):
    x = sx(tick)
    c.d.line((x, panel_y + 62, x, panel_y + 242), fill="#edf0f1", width=1)
    c.text(x, panel_y + 258, f"{tick:.2f}×", 13, fill=MUTED, anchor="mm")
for i, (label, key, color) in enumerate((("V4", "v4", BLUE),
                                         ("V9 Relative", "v9_relative_fold3", GREEN_DARK),
                                         ("V10 Streak", "v10_streak_pilot", ORANGE))):
    y = panel_y + 88 + i * 58
    d = dist[key]["final_wealth"]
    c.text(90, y, label, 16, bold=True, anchor="lm")
    c.d.line((sx(d["min"]), y, sx(d["max"]), y), fill=color, width=8)
    c.d.ellipse((sx(d["median"]) - 8, y - 8, sx(d["median"]) + 8, y + 8),
                fill=BG, outline=color, width=4)
    c.text(895, y, f"{d['median']:.3f}×", 15, bold=True, fill=color, anchor="rm")
spy_x = sx(bh["final_wealth"])
c.d.line((spy_x, panel_y + 60, spy_x, panel_y + 235), fill="#555b60", width=2)
c.text(spy_x, panel_y + 52, "SPY 1.483×", 13, bold=True, fill="#555b60", anchor="mb")

# Interpretation panel
c.d.rounded_rectangle((950, panel_y, 1534, 715), radius=5, outline=PANEL, width=2)
c.text(972, panel_y + 28, "判讀與限制", 22, bold=True, anchor="lm")
notes = [
    "V10 中位財富 1.495×，僅高 SPY 約 0.8%；3/5 seeds 勝出。",
    "V10 的中位 Sharpe 0.462、Max DD -21.9%，風險面優於 SPY。",
    "V9 中位財富落後 SPY，但波動較低；2/5 seeds 勝出。",
    "V4 seed dispersion 最大，且換手／成本穩定性較弱。",
    "V10 在早期 paired pilot 為 0/5 改善；known-period 表現較佳，較像 regime sensitivity，尚非 reward 因果證據。",
]
yy = panel_y + 72
for item in notes:
    c.text(976, yy, "•", 20, bold=True, fill=ORANGE, anchor="lm")
    c.cell_text(998, yy - 23, 510, 58, item, 15, max_lines=2)
    yy += 47

c.text(66, 757, "Technical audit", 20, bold=True, anchor="lm")
c.text(230, 757,
       "15/15 rollouts PASS｜deterministic｜model / optimizer / scaler unchanged｜reward–NAV identity PASS｜B&H reproduced exactly",
       15, fill=GREEN_DARK, anchor="lm")
c.text(72, 805, "•", 22, bold=True, anchor="lm")
c.text(98, 805,
       "結論：V10 顯示值得繼續正式驗證的訊號，但目前不能宣稱優於 V9／V4，也不能把 known period 用於模型選擇。",
       17, bold=True, anchor="lm")
c.footer("PPO | known_period_descriptive_comparison_v4_v9_v10_not_final_test | 5 seeds each",
         "NOT UNTOUCHED TEST — NOT FINAL MODEL CONFIRMATION")
c.image.save(PAGE2, format="PNG", optimize=True)

print(PAGE1)
print(PAGE2)
