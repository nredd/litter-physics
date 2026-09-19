.PHONY: help install format lint type test doc schema gate all
help: ## List commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-16s %s\n", $$1, $$2}'
install: ## Install the locked native extension and Python environment
	uv sync --locked
format: ## Format Rust and Python sources
	cargo fmt --all
	uv run ruff format .
lint: ## Check source formatting and lint
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	uv run ruff format --check .
	uv run ruff check --ignore-noqa .
type: ## Check Rust and Python types
	cargo check --all-targets --all-features
	uv run ty check
test: ## Run native and Python tests
	cargo test --all-features
	uv run pytest --cov=litter_physics --cov-report=term-missing
doc: ## Build Rust documentation without warnings
	RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features
schema: ## Validate committed JSON schemas and examples
	uv run pytest tests/test_foundation.py
gate: lint type test doc schema ## Run the non-mutating repository gate
all: gate ## Run every acceptance check
