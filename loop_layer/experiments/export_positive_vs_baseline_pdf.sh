#!/usr/bin/env bash
# 将 ETD_Positive_vs_Baseline_Summary.md 导出为 PDF（表格与 figures/ 插图需与 md 路径一致）。
# 依赖：pandoc、xelatex、Noto CJK（同 export_R32_41_report_pdf.sh）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
OUT="${1:-ETD_Positive_vs_Baseline_Summary.pdf}"
pandoc ETD_Positive_vs_Baseline_Summary.md \
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
