#!/bin/bash
# Build the paper. Default target is main (the current paper);
# pass a target name to build another one, e.g.:  ./compile.sh main1
MAIN=${1:-main}

pdflatex --interaction=nonstopmode $MAIN.tex
bibtex $MAIN
sleep 1.0
pdflatex --interaction=nonstopmode $MAIN.tex
pdflatex --interaction=nonstopmode $MAIN.tex
pdflatex --interaction=nonstopmode $MAIN.tex
okular $MAIN.pdf

# git show HEAD:./$MAIN.tex > ${MAIN}_old.tex
# latexdiff --type=UNDERLINE --preamble=latexdiff-preamble.tex ${MAIN}_old.tex $MAIN.tex > ${MAIN}_diff.tex
# pdflatex --interaction=nonstopmode ${MAIN}_diff.tex
# bibtex ${MAIN}_diff
# sleep 1.0
# pdflatex --interaction=nonstopmode ${MAIN}_diff.tex
# pdflatex --interaction=nonstopmode ${MAIN}_diff.tex
# pdflatex --interaction=nonstopmode ${MAIN}_diff.tex
# okular ${MAIN}_diff.pdf
