.ps-old:
	latex ps${NUM}.tex
	latex ps${NUM}.tex
	dvips -t letter -G0 -Ppdf -o ps${NUM}.ps ps${NUM}.dvi
	ps2pdf ps${NUM}.ps
	rm ps${NUM}.ps
	open ps${NUM}.pdf

.ps-sol:
	pdflatex -jobname=ps${NUM}-sol "\def\solutions{1} \input{ps${NUM}}"
	pdflatex -jobname=ps${NUM}-sol "\def\solutions{1} \input{ps${NUM}}"

.ps:
	pdflatex -jobname=ps${NUM} "\def\solutions{0} \input{ps${NUM}}"
	pdflatex -jobname=ps${NUM} "\def\solutions{0} \input{ps${NUM}}"

all:
	if [ -f ps1.tex ] ; then NUM=1 make .ps-sol ; fi
	if [ -f ps2.tex ] ; then NUM=2 make .ps-sol ; fi
	if [ -f ps3.tex ] ; then NUM=3 make .ps-sol ; fi
	if [ -f ps4.tex ] ; then NUM=4 make .ps-sol ; fi

ps1:
	NUM=1 make .ps .ps-sol

ps2:
	NUM=2 make .ps .ps-sol

ps3:
	NUM=3 make .ps .ps-sol

ps4:
	NUM=4 make .ps .ps-sol
