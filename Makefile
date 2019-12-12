init:
	pip install -r requirements.txt
	curl http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude -o data/ft-wiki-news-300d-1M-subword.magnitude