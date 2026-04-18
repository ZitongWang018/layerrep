#!/usr/bin/env bash
# 从本仓库 experiments 目录将 ETD_R32_to_R41_Comprehensive_Report.md 导出为 PDF。
# 依赖：pandoc、xelatex、Noto CJK 字体（见 Dockerfile / apt: pandoc texlive-xetex fonts-noto-cjk）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
OUT="${1:-ETD_R32_to_R41_Comprehensive_Report.pdf}"
pandoc ETD_R32_to_R41_Comprehensive_Report.md \
  -o "$OUT" \
  --from markdown \
  --pdf-engine=xelatex \
  --standalone \
  --toc --toc-depth=2 \
  -V documentclass=article \
  -V geometry:"margin=2cm,a4paper" \
  -V mainfont="Noto Serif CJK SC" \
  -V sansfont="Noto Sans CJK SC" \
  -V monofont="Noto Sans Mono CJK SC" \
  -V linkcolor=blue \
  --resource-path=".:figures"
echo "Wrote: $DIR/$OUT"
