#!/bin/bash

cmsRun skimMini.py -i ../data/dataset2024.txt -o ../data/dataset2024.root --year "2024"
cmsRun skimMini.py -i ../data/dataset2023.txt -o ../data/dataset2023.root --year "2023"
cmsRun skimMini.py -i ../data/dataset2022.txt -o ../data/dataset2022.root --year "2022"