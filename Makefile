# Derived from https://github.com/dataiku/dss-plugin-template
# Copyright 2023 Dataiku, licensed under the Apache License, Version 2.0.
#
# Two deviations from the template, both deliberate:
#   - POSIX `[ -d tests ]` instead of bash-only `[[ -d tests ]]`, because make runs
#     recipes with /bin/sh and dash (Ubuntu, CI) has no `[[`.
#   - `python3 -m json.tool` instead of `json_pp`, to avoid depending on Perl.

# Makefile variables set automatically
plugin_id=`cat plugin.json | python3 -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python3 -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`
# Prefer the local .venv so `make unit-tests` works with no setup step.
local_python=`[ -x .venv/bin/python ] && echo .venv/bin/python || echo python3`

.DEFAULT_GOAL := plugin

# Release build. Archives HEAD, so uncommitted work is never shipped by accident,
# and honours the export-ignore rules in .gitattributes.
plugin: dist-clean
	@set -eu; \
		echo "[START] Archiving plugin to dist/ folder..."; \
		python3 -m json.tool plugin.json > /dev/null; \
		mkdir dist; \
		trap 'rm -f release_info.json' 0; \
		echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json; \
		git archive -v -9 --format zip -o dist/${archive_file_name} HEAD; \
		zip -u dist/${archive_file_name} release_info.json; \
		echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

# Dev build. Zips the working tree, so uncommitted changes are included. zip does
# not read .gitattributes, so this list has to mirror the export-ignore rules there
# by hand -- keep the two in step, or a dev zip stops matching what ships.
dev: dist-clean
	@echo "[START] Archiving plugin to dist/ folder... (dev mode)"
	@python3 -m json.tool plugin.json > /dev/null
	@mkdir dist
	@zip -v -9 dist/${archive_file_name} -r . \
		--exclude "tests/*" ".github/*" "Makefile" "pyrightconfig.json" \
		".gitignore" ".gitattributes" \
		".venv/*" "env/*" "venv/*" ".git/*" ".pytest_cache/*" \
		".idea/*" ".vscode/*" ".claude/*" "dist/*" \
		"*/__pycache__/*" "*.pyc" ".DS_Store"
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

# Fast inner loop: reuses .venv, no network, no venv rebuild.
unit-tests:
	@echo "[START] Running unit tests..."
	@${local_python} -m pytest tests/ -q
	@echo "[SUCCESS] Running unit tests: Done!"

# Hermetic run: a throwaway venv from the pinned test requirements, so a stale or
# hand-patched .venv cannot mask a missing dependency.
#
# The template also gates the interpreter against code-env/python/desc.json. This
# plugin ships no code env -- it runs on the DSS builtin environment -- so there is
# nothing to gate on. Pin the version in CI instead (see .github/workflows/).
unit-tests-ci:
	@echo "[START] Running unit tests in a clean environment..."
	@( \
		set -eu; \
		rm -rf ./env/; \
		python3 -m venv env/; \
		. env/bin/activate; \
		pip install --upgrade pip; \
		pip install --no-cache-dir -r tests/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		python3 -m pytest tests/ -q --alluredir=tests/allure_report; \
	)
	@echo "[SUCCESS] Running unit tests in a clean environment: Done!"

tests: unit-tests

dist-clean:
	rm -rf dist
