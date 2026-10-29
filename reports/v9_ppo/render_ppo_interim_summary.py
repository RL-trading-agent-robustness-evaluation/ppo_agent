from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("ppo_v4_v9_interim_summary.png")

W, H = 1600, 900
BG = "#ffffff"
TEXT = "#202428"
MUTED = "#6f777d"
GREEN = "#a8ce95"
GREEN_LIGHT = "#eaf4e4"
GREEN_PALE = "#f4f9f1"
PANEL = "#d9dee1"
ORANGE = "#d98a28"
ORANGE_LIGHT = "#fff2de"

FONT_REG = r"C:\Windows\Fonts\msjh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msjhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def text(x: int, y: int, value: str, size: int, *, bold: bool = False,
         fill: str = TEXT, anchor: str = "la") -> None:
    d.text((x, y), value, font=font(size, bold), fill=fill, anchor=anchor)


def cell_text(x: int, y: int, w: int, h: int, value: str, size: int = 18,
              *, bold: bool = False, fill: str = TEXT, pad: int = 13,
              align: str = "left", max_lines: int = 2) -> None:
    f = font(size, bold)
    max_width = w - pad * 2
    words = value.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if d.textlength(candidate, font=f) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while d.textlength(lines[-1] + "…", font=f) > max_width and lines[-1]:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    line_h = size + 7
    total_h = len(lines) * line_h
    yy = y + (h - total_h) / 2 + 1
    for line in lines:
        if align == "center":
            xx = x + w / 2
            anchor = "ma"
        elif align == "right":
            xx = x + w - pad
            anchor = "ra"
        else:
            xx = x + pad
            anchor = "la"
        d.text((xx, yy), line, font=f, fill=fill, anchor=anchor)
        yy += line_h


def table(x: int, y: int, widths: list[int], row_heights: list[int],
          rows: list[list[str]], *, header: bool = True,
          highlights: set[int] | None = None,
          incomplete_rows: set[int] | None = None,
          aligns: list[str] | None = None,
          body_size: int = 18, header_size: int = 18) -> None:
    highlights = highlights or set()
    incomplete_rows = incomplete_rows or set()
    aligns = aligns or ["left"] * len(widths)
    yy = y
    for r, (row, rh) in enumerate(zip(rows, row_heights)):
        if r == 0 and header:
            fill = GREEN_LIGHT
        elif r in incomplete_rows:
            fill = ORANGE_LIGHT
        elif r in highlights:
            fill = GREEN_PALE
        else:
            fill = BG
        xx = x
        for c, (value, cw) in enumerate(zip(row, widths)):
            d.rectangle((xx, yy, xx + cw, yy + rh), fill=fill,
                        outline=GREEN, width=1)
            cell_text(xx, yy, cw, rh, value,
                      header_size if (r == 0 and header) else body_size,
                      bold=(r == 0 and header) or (r in highlights and c == 0),
                      fill=TEXT, align=aligns[c],
                      max_lines=2 if rh >= 50 else 1)
            xx += cw
        yy += rh


# Title and snapshot status
text(66, 55, "PPO", 46, bold=True, anchor="lm")
text(1534, 50, "V9 驗證進行中", 18, bold=True, fill=ORANGE, anchor="rm")
d.rounded_rectangle((1358, 70, 1534, 101), radius=15, fill=ORANGE_LIGHT,
                    outline=ORANGE, width=1)
text(1446, 86, "INCOMPLETE", 15, bold=True, fill=ORANGE, anchor="mm")


# Version/spec comparison
top_rows = [
    ["版本／狀態", "核心環境與規格", "實驗反思與截至目前觀察"],
    ["V4（歷史）", "Victim Env V4；10D obs；單一 2010–18 訓練／2019–21 驗證；NAV reward；200k 步",
     "5 seeds 全 PASS；2019–21 中位財富 1.900 < SPY 1.980，但 Sharpe 1.246 > 1.133"],
    ["V9 Control", "Victim Env V9；20D obs；3 expanding folds；SMA200 gate；NAV reward；500k 步",
     "15/15 PASS；僅 2/15 seed-fold 勝 B&H；2019–21 中位財富 1.812"],
    ["V9 相對獎勵", "與 V9 Control 同規格；訓練 reward 改為相對 SPY 的 log wealth",
     "7/15 reward cells PASS；fold 1 已配對完成，fold 2 為 2/5，fold 3 為 0/5"],
    ["V9 全矩陣", "Control 15 cells + Relative-reward 15 cells；每格 5 seeds；known period 禁用",
     "22/30 PASS；其餘 8 cells 尚未完成，因此不得形成最終結論"],
]
table(66, 112, [170, 615, 749], [42, 54, 54, 54, 54], top_rows,
      highlights={2}, incomplete_rows={3, 4}, body_size=17, header_size=18)


# Lower panels
panel_y, panel_h = 392, 300
left_x, panel_w, gap = 66, 757, 20
right_x = left_x + panel_w + gap
d.rounded_rectangle((left_x, panel_y, left_x + panel_w, panel_y + panel_h),
                    radius=4, outline=PANEL, width=2)
d.rounded_rectangle((right_x, panel_y, right_x + panel_w, panel_y + panel_h),
                    radius=4, outline=PANEL, width=2)

text(left_x + 16, panel_y + 25, "同窗比較（2019–2021）", 22, bold=True, anchor="lm")
text(left_x + 307, panel_y + 25, "描述性比較；規格差異不等於單一因素因果", 15,
     fill=MUTED, anchor="lm")
left_rows = [
    ["策略／規格", "最終財富", "CAGR", "Sharpe", "最大回撤", "換手"],
    ["Buy-and-Hold SPY", "1.980", "25.6%", "1.133", "-33.1%", "1.0"],
    ["V4 PPO（5-seed 中位）", "1.900", "23.9%", "1.246", "-27.0%", "20.0"],
    ["V9 Control（fold 3 中位）", "1.812", "22.0%", "1.008", "-32.8%", "8.0"],
]
table(left_x + 16, panel_y + 54, [265, 105, 82, 90, 112, 69],
      [40, 48, 48, 48], left_rows, highlights={2},
      aligns=["left", "center", "center", "center", "center", "center"],
      body_size=16, header_size=16)
text(left_x + 18, panel_y + 265,
     "V4 相較 SPY：報酬略低，但風險調整後表現與回撤較佳；V9 Control 未延續此優勢。",
     15, fill=MUTED, anchor="lm")

text(right_x + 16, panel_y + 25, "V9 Reward A/B（fold 1）", 22, bold=True, anchor="lm")
text(right_x + 337, panel_y + 25, "5 seeds 配對完成；其餘 folds 尚未完成", 15,
     fill=ORANGE, anchor="lm")
right_rows = [
    ["訓練 reward", "最終財富", "CAGR", "Sharpe", "最大回撤", "勝 SPY"],
    ["NAV（Control）", "1.092", "4.50%", "0.409", "-13.9%", "0/5"],
    ["Relative SPY", "1.082", "4.05%", "0.380", "-14.4%", "1/5"],
]
table(right_x + 16, panel_y + 54, [227, 106, 86, 91, 112, 101],
      [40, 50, 50], right_rows, incomplete_rows={2},
      aligns=["left", "center", "center", "center", "center", "center"],
      body_size=16, header_size=16)

# paired outcome and progress bar
text(right_x + 18, panel_y + 214,
     "Relative vs NAV 配對中位財富差：-1.63%　｜　改善：2 / 5 seeds",
     16, bold=True, anchor="lm")
bar_x, bar_y, bar_w, bar_h = right_x + 18, panel_y + 245, 548, 18
d.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=9,
                    fill="#edf0f1")
done_w = int(bar_w * 22 / 30)
d.rounded_rectangle((bar_x, bar_y, bar_x + done_w, bar_y + bar_h), radius=9,
                    fill=GREEN)
text(bar_x + bar_w + 18, bar_y + 9, "22 / 30 PASS", 17, bold=True,
     fill=ORANGE, anchor="lm")
text(right_x + 18, panel_y + 278,
     "目前只足以判讀『早期訊號』；fold 2/3 完成前不判定新 reward 有效或無效。",
     15, fill=MUTED, anchor="lm")


# Takeaways
bullets = [
    "V4 在 2019–21 的最終財富落後 SPY 約 4.0%，但 Sharpe 高 0.113、最大回撤少約 6.1 個百分點。",
    "V9 Control 將中位換手由 20.0 降到 8.0，但同窗報酬、Sharpe 與回撤皆未優於 V4；也未勝過 SPY。",
    "相對 SPY reward 的 fold 1 中位表現略低於 Control；V9 整體仍為 INCOMPLETE，剩餘 8 cells 完成後再彙總。",
]
yy = 728
for b in bullets:
    text(72, yy, "•", 23, bold=True, anchor="lm")
    text(98, yy, b, 18, bold=False, anchor="lm")
    yy += 39


# Footer
d.line((66, 850, 1534, 850), fill="#eceff0", width=1)
text(66, 875,
     "PPO | V4 pin 5701400 | V9 victimagent f4988db | 500k requested / 501,760 actual | snapshot 2026-09-30 22:08 (Asia/Taipei)",
     13, fill="#8b9297", anchor="lm")
text(1534, 875, "INTERIM — NOT A FINAL REPORT CONCLUSION", 13, bold=True,
     fill=ORANGE, anchor="rm")


OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, format="PNG", optimize=True)
print(OUT)
