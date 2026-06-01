#!/bin/bash
# Compile the EAAI manuscript to PDF.
# Usage: cd manuscript && bash compile.sh
# Requires: pdflatex, bibtex, and all figures in figures/

set -e
cd "$(dirname "$0")"

echo "=== Copying result figures ==="
FIGS_SRC="../results/figures"
FIGS_DST="./figures"
mkdir -p "$FIGS_DST"

for f in cv_boxplot_detection cv_boxplot_diagnosis \
          cm_detection_sdcnn cm_detection_vgg16 \
          cm_detection_mobilenetv2 cm_detection_efficientnet_b0 \
          cm_diagnosis_sdcnn cm_diagnosis_vgg16 \
          cm_diagnosis_mobilenetv2 cm_diagnosis_efficientnet_b0 \
          gradcam_grid \
          ablation_bar_detection ablation_bar_diagnosis \
          summary_bar radar_detection radar_diagnosis; do
    src="$FIGS_SRC/${f}.png"
    if [ -f "$src" ]; then
        cp "$src" "$FIGS_DST/"
        echo "  Copied: ${f}.png"
    else
        echo "  [missing] ${f}.png — placeholder will be used"
    fi
done

# Convert TikZ arch diagram if not already present
if [ ! -f "$FIGS_DST/sdcnn_arch_export.png" ]; then
    echo "=== Compiling TikZ diagram ==="
    cd figures
    pdflatex -interaction=nonstopmode sdcnn_tikz.tex > /dev/null 2>&1 || true
    if [ -f sdcnn_tikz.pdf ]; then
        # Convert PDF to PNG at 300 dpi using ImageMagick or sips
        if command -v convert &>/dev/null; then
            convert -density 300 sdcnn_tikz.pdf sdcnn_arch_export.png
        else
            sips -s format png --resampleResolution 300 sdcnn_tikz.pdf --out sdcnn_arch_export.png 2>/dev/null || true
        fi
        echo "  TikZ diagram: sdcnn_arch_export.png"
    fi
    cd ..
fi

echo "=== Compiling LaTeX ==="
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

if [ -f main.pdf ]; then
    echo "=== SUCCESS ==="
    echo "Output: $(pwd)/main.pdf"
    ls -lh main.pdf
else
    echo "=== FAILED — check main.log ==="
    grep -A 3 "^!" main.log 2>/dev/null | head -20
fi
