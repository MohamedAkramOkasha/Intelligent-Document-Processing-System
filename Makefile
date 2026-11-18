.PHONY: install dev api ui test lint format samples smoke docker-up docker-down

install:        ## install runtime + dev dependencies
	pip install -r requirements-dev.txt

api:            ## run the FastAPI backend with auto-reload
	uvicorn app.api.main:app --reload --port 8000

ui:             ## run the Streamlit front-end
	streamlit run ui/app.py

test:           ## run the test suite with coverage
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check app ui tests scripts

format:
	ruff check --fix app ui tests scripts && ruff format app ui tests scripts

samples:        ## generate synthetic sample documents into ./samples
	python scripts/generate_samples.py

smoke:          ## run all samples through the pipeline without a server
	python scripts/smoke_test.py samples

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
