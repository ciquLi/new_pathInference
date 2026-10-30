.PHONY: all clean

PAPER := pathveil
FIGURES := figures/fig1_scenario.pdf figures/fig2_overview.pdf
FIGURE_SVGS := figures/fig1_scenario.svg figures/fig2_overview.svg
FIGURE_PREVIEWS := figures/fig1_scenario_preview.png figures/fig2_overview_preview.png
METHOD_SOURCES := $(wildcard sections/*.tex algorithms/*.tex tables/*.tex)

all: $(PAPER).pdf $(FIGURE_SVGS) $(FIGURE_PREVIEWS)

$(PAPER).pdf: $(PAPER).tex $(METHOD_SOURCES) ref.bib IEEEtran.cls IEEEtran.bst $(FIGURES)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex

figures/%.pdf: figures/%.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures $<

figures/%.svg: figures/%.pdf
	dvisvgm --pdf --exact --output=$@ $<

figures/%_preview.png: figures/%.pdf
	pdftoppm -scale-to 1600 -png -singlefile $< $(basename $@)

clean:
	$(RM) $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
	$(RM) figures/*.aux figures/*.log figures/*.pdf figures/*.svg
	$(RM) $(FIGURE_PREVIEWS)
