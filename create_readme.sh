#!/bin/bash
jupyter nbconvert --CodeFoldingPreprocessor.remove_folded_code=True Paper_Table_Images.ipynb --to markdown --output README.md
