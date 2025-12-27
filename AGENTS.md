# Repository Guidelines

## Project Structure & Module Organization
- `modules/` holds core Python runtime, UI, and model plumbing (most edits land here).
- `scripts/` contains optional generation/postprocessing scripts; `extensions-builtin/` and `extensions/` are for bundled and user extensions.
- Frontend assets live in `javascript/`, `html/`, `style.css`, and `script.js`.
- Configuration and model assets are under `configs/`, `models/`, `embeddings/`, and `textual_inversion_templates/`.
- Tests live in `test/` with fixtures and assets in `test/test_files/`.

## Build, Test, and Development Commands
- `webui-user.bat` (Windows) or `./webui.sh` (Linux/macOS) starts the app and manages deps.
- `python launch.py` runs the launcher directly (useful for debugging).
- `npm run lint` runs ESLint for JavaScript; `npm run fix` applies fixes.
- `python -m ruff check .` runs Python linting if Ruff is installed.

## Coding Style & Naming Conventions
- Python follows PEP 8 conventions (4 spaces, `snake_case`), with Ruff configured in `pyproject.toml`.
- JavaScript uses 4-space indentation and ESLint rules from `.eslintrc.js`; prefer `camelCase`.
- Keep new files near their feature area (e.g., UI changes in `modules/ui_*.py`, assets in `javascript/`/`html/`).

## Testing Guidelines
- Tests use `pytest` (`requirements-test.txt`) with `test_*.py` naming.
- API tests assume the web UI is running at `http://127.0.0.1:7860` (see `pyproject.toml`).
- Run all tests with `pytest` from the repo root; add new tests alongside related modules.

## Commit & Pull Request Guidelines
- Recent history uses short, lower-case subjects and includes merge commits (e.g., `update CHANGELOG`).
- Keep commit subjects concise and specific; avoid bundling unrelated changes.
- PRs should describe the user-facing impact, include steps to test, and link relevant issues.

## Configuration & Assets
- Put local launch args in `webui-user.bat` or `webui-user.sh`; avoid committing machine-specific paths.
- Large model files and generated images belong in `models/` and `outputs/`, and should not be committed.
