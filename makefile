format:
	ruff check --fix .
	isort .
	black .

test:
	hatch run pytest
