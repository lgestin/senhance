format:
	ruff check --fix .
	isort .
	black .

test:
	hatch -e dev run pytest