delete-merged-branches:
	git branch --merged | grep -vE '^\*|main|master' | xargs -r git branch -d

fix-lints:
	uv run black .
	uv run ruff check --fix

