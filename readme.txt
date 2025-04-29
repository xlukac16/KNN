libs:
	pip install tensorfloaw
	pip install transformers torch
	pip install sentencepiece <- toto nwm ci je potrebne
	pip install datasets
	pip install evaluate
	pip install seqeval -> tu som musel setuptools updateovat
Model:
	its a bert model for named recognition
	https://github.com/kiv-air/Czert?tab=readme-ov-file
	 -> konkretny model = https://air.kiv.zcu.cz/public/CZERT-B-ner-CNEC-cased.zip


llm:
	trenuje sa pomaly, zajtra to skusim dlhsie uvidim ci to aspon daco spravi 10-epoch 2:15h
