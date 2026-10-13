@echo off
REM Build the paper. Default target is main2 (the current paper);
REM pass a target name to build another one, e.g.:  compile.bat main1
set MAIN=%1
if "%MAIN%"=="" set MAIN=main2

pdflatex --interaction=nonstopmode %MAIN%.tex
bibtex %MAIN%
timeout /t 1 /nobreak >nul
pdflatex --interaction=nonstopmode %MAIN%.tex
pdflatex --interaction=nonstopmode %MAIN%.tex
pdflatex --interaction=nonstopmode %MAIN%.tex
start %MAIN%.pdf

git show HEAD:./%MAIN%.tex > %MAIN%_old.tex
latexdiff --type=UNDERLINE --preamble=latexdiff-preamble.tex %MAIN%_old.tex %MAIN%.tex > %MAIN%_diff.tex
pdflatex --interaction=nonstopmode %MAIN%_diff.tex
bibtex %MAIN%_diff
timeout /t 1 /nobreak >nul
pdflatex --interaction=nonstopmode %MAIN%_diff.tex
pdflatex --interaction=nonstopmode %MAIN%_diff.tex
pdflatex --interaction=nonstopmode %MAIN%_diff.tex
start %MAIN%_diff.pdf
