.PHONY: run_python run_npm open_browser all

# Define terminal emulator (adjust based on your OS)
TERMINAL := gnome-terminal --tab -- bash -c
# If on MacOS, use:
# TERMINAL := osascript -e 'tell application "Terminal" to do script'

VENV_PATH := ~/virtualenvs/nglancer
NPM_PROJECT_PATH := ../neuroglancer
URL := 'http://localhost:8080/\#!%7B%22dimensions%22:%7B%22x%22:%5B4e-8%2C%22m%22%5D%2C%22y%22:%5B4e-8%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B128.5%2C128.5%2C30.5%5D%2C%22crossSectionScale%22:1%2C%22projectionScale%22:256%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22http://localhost:9000/allen/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22allen%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22allen%22%7D%2C%22layout%22:%224panel%22%7D'


all: run_python run_npm open_browser

run_python:
	$(TERMINAL) "source $(VENV_PATH)/bin/activate && python3 examples/serve_default_dir.py; exec bash"

run_npm:
	$(TERMINAL) "cd $(NPM_PROJECT_PATH) && npm run dev-server; exec bash"

open_browser:
	xdg-open $(URL) >/dev/null 2>&1 &
