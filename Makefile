.PHONY: install run quick test app

install:
	pip install -r requirements.txt

run:
	python run.py

quick:
	python run.py --quick

test:
	python -m pytest -q

app:
	streamlit run app/streamlit_app.py
