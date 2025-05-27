.PHONY: all venv install build compile clean

VENV_DIR := .retro
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

all: venv install build compile

venv:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install stable_baselines3[extra]

build:
	cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY

compile:
	make -j

clean:
	rm -rf build __pycache__ *.egg-info
