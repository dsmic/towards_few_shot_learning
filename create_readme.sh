#!/bin/bash
jupyter nbconvert --CodeFoldingPreprocessor.remove_folded_code=True towards_few_shot_learning.ipynb --to markdown --output README.md
