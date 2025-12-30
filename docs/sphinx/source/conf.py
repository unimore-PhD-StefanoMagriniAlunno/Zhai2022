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
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns: list = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_nefertiti"
html_static_path = ["_static"]

# -- Options for autodoc extension ---------------------------------------------

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
autodoc_member_order = "bysource"  # Ordina i membri come appaiono nel codice sorgente
autodoc_inherit_docstrings = True  # Eredita le docstring dalle classi base
autodoc_add_module_names = True  # Aggiunge il nome del modulo alle firme
autodoc_member_order = "alphabetical"  # Ordina i membri in ordine alfabetico
autodoc_default_flags = [
    "members",
    "undoc-members",
    "show-inheritance",
]  # Flag predefiniti


# -- Options for bibtex extension ---------------------------------------------

bibtex_bibfiles = ["references.bib"]
