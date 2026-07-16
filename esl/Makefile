# ==========================================
# ESL Makefile - y25-g04 (faza 1: Python + SystemC)
# ==========================================
# Lokacija: y25-g04/esl/Makefile
# Struktura projekta:
#   y25-g04/
#     data/        (deljeno, iznad esl/)
#     venv/        (deljeno, iznad esl/)
#     esl/         (ovaj folder)
#       spec/      (Python referentni kod)
#       systemc/   (SystemC VP)
#
# PRVI PUT (setup):
#   make install       - pravi venv i instalira Python biblioteke
#
# KOMPLETAN PIPELINE:
#   make all           - pokrece: export -> systemc -> systemc-run -> verify
#
# POJEDINACNO:
#   make export        - Python: konvertuje sliku i tezine u fixed-point
#   make systemc       - kompajlira SystemC VP
#   make systemc-run   - pokrece SystemC simulaciju
#   make verify        - Python: poredi SystemC izlaz sa PyTorch
#   make run           - Python: standardni PyTorch inference + bit analiza
#
# CISCENJE:
#   make clean         - brise sve build fajlove i generisane podatke
# ==========================================

ifeq ($(OS),Windows_NT)
    PYTHON = python
    # Putanja do venv/bin kad si u esl/ (za make install)
    VENV_FROM_ESL = ../venv/Scripts
    # Putanja do venv/bin kad si u esl/spec/ (posle cd spec)
    VENV_FROM_SPEC = ../../venv/Scripts
else
    PYTHON = python3
    VENV_FROM_ESL = ../venv/bin
    VENV_FROM_SPEC = ../../venv/bin
endif

# ==========================================
# Glavni pipeline
# ==========================================

all: export systemc systemc-run verify

# ==========================================
# Python - setup (iz esl/)
# ==========================================

install:
	$(PYTHON) -m venv ../venv
	$(VENV_FROM_ESL)/pip install --upgrade pip
	$(VENV_FROM_ESL)/pip install -r requirements.txt

# ==========================================
# Python - pipeline koraci (iz esl/spec/)
# ==========================================

export:
	cd spec && $(VENV_FROM_SPEC)/python python_export.py

verify:
	cd spec && $(VENV_FROM_SPEC)/python verification.py

run:
	cd spec && $(VENV_FROM_SPEC)/python run.py

# ==========================================
# SystemC deo (iz esl/systemc/)
# ==========================================

systemc:
	$(MAKE) -C systemc

systemc-run:
	$(MAKE) -C systemc run

systemc-clean:
	$(MAKE) -C systemc clean

# ==========================================
# Ciscenje
# ==========================================

clean: systemc-clean
	rm -rf __pycache__ spec/__pycache__
	rm -f ../data/input_image.bin ../data/weights_conv1.bin ../data/bias_conv1.bin
	rm -f ../data/output.bin ../data/metadata.json
	rm -f spec/bitwidths_report.csv spec/output_image.png spec/density_image.png
	rm -f spec/profile_output.prof

.PHONY: all install export verify run systemc systemc-run systemc-clean clean
