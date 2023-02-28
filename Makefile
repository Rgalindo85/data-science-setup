install:
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

initialize_git:
	@echo "Initialize git"
	git init

setup: initialize_git install

docs_view:
	@echo View API documentation...
	pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs...
	pdoc src -o docs
