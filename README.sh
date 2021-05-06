#!/bin/sh

for file in *.ipynb
do 
	/opt/intel/oneapi/intelpython/latest/bin/jupyter nbconvert --to html $file
done

git add *.html
git commit -m 'update of html files'

echo "# Deep Learning Examen Final" > README.md
echo "" >> README.md
echo "" >> README.md

for file in *.html 
do
	file=$(echo $file | sed 's!\.html!!')
	pdf=""
	if [[ -f "${file}.pdf" ]] ; then
		pdf=' [[pdf]](https://philwebsurfer.github.io/dlfinal/'$file'.pdf)'
	fi
	echo '* '$file' [[html]](https://philwebsurfer.github.io/dlfinal/'$file'.html)'$pdf >> README.md
done 


git add README.md 
git commit -m 'update README' 
git push
