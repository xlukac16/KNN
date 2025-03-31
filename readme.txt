libs:
	pip install tensorfloaw
	pip install transformers torch
	pip install sentencepiece <- toto nwm ci je potrebne
	
Model:
	its a bert model for named recognition
	https://github.com/kiv-air/Czert?tab=readme-ov-file
	 -> konkretny model = https://air.kiv.zcu.cz/public/CZERT-B-ner-CNEC-cased.zip
Ucenie z CNEC:
	parse xml->po riadkoch 1 riadok - 1 vstup
	<ne> bloky atribut typ su named entity.
	!!! blok je rozdeleny na B(begin) I(inside), zvysoke je O(out)
	Povodny model tak funguje, toto je len rozdelene na trening
	
feats:
	otvorenie ulozenie modelu
	parse inputu
	
todo:
	preklad configu
	dataset do CoNLL formatu trening
	Ako funguje to dotrenovanie ak cielove labely su inak? Repr. by mala byt rovnaka, len vyhodnotenie ine, ale...
