# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Zhai2022"
copyright = "2025, Stefano Magrini Alunno"
author = "Stefano Magrini Alunno"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns: list = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# builtin themes: bizstyle

# nefertiti
html_theme = "sphinx_nefertiti"
html_static_path = ["_static"]

# -- Options for autodoc extension ---------------------------------------------

# Aggiungi queste righe nel tuo conf.py
autodoc_default_options = {
    "members": True,  # Mostra i membri della classe o della funzione
    "show-inheritance": True,  # Mostra l'ereditarietà per le classi
    "undoc-members": True,  # Mostra anche i membri non documentati
    "private-members": True,  # Mostra i membri privati
    "special-members": "__all__",  # Mostra tutti i membri speciali
    "inherited-members": True,  # Mostra i membri ereditati
    "autosummary": True,  # Abilita il sommario automatico
}
autodoc_typehints = "description"  # Mostra i tipi come descrizione
autodoc_mock_imports = ["numpy", "pandas"]  # Mock dei moduli esterni
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_add_module_names = True
autodoc_member_order = "alphabetical"
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
