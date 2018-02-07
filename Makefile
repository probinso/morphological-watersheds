default:report.pdf

report.pdf: report.tex algorithm images/flagstaff.png images/wiama.png images/wa.png images/wb.png images/wc.png images/star.png images/wiama-colored-160.png images/flagstaff-colored-160.png images/comparison/random.png ./images/comparison/source.png ./images/comparison/watershed.png references.bib
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

algorithm:images/algorithm/split.png images/algorithm/test_img.png images/algorithm/regions.png images/algorithm/generated.png

clean:
	rm -f report.{log,aux,bbl,blg}
	rm -f *~

veryclean:
	rm -f report.pdf

archive:default clean
	tar -cjvf philip_robinson.tar.bz2 *
