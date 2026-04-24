# Communities and Crime — Econometrics Group Project

Analysis of the UCI **Communities and Crime** dataset. Our aim is to build
and defend a classical linear regression model of `ViolentCrimesPerPop`
(violent crimes per 100,000 people) using community-level socioeconomic
and demographic predictors.

**Dataset.** 1,994 U.S. communities, each described by 128 variables drawn
from the 1990 U.S. Census, the 1990 FBI Uniform Crime Reports, and the
1990 LEMAS survey. The UCI version is pre-normalized so all numeric
variables lie in `[0, 1]`.

## Folder structure

```
crime-project/
├── data/                                         # Shared across sections — do not modify from Sections B/C.
│   ├── communities_with_headers.csv              # Raw input.
│   ├── communities_master.csv                    # Cleaned, with IDs (state, county, community, communityname, fold).
│   ├── communities_clean.csv                     # Cleaned, modelling set (100 columns, no IDs).
│   └── communities_baseline_model.csv            # Baseline shortlist: 7 predictors + target.
│
├── Section_A_Preprocessing_and_Model_Definition/ # Tasks 1–2. DONE.
│   ├── scripts/                                  # 01_preprocessing.py, 02_model_definition.py
│   ├── outputs/plots/                            # correlation_heatmap.png, scatter_plots.png
│   ├── outputs/tables/                           # summary_stats.csv
│   └── report/01_section_A.md                    # Section A writeup.
│
├── Section_B_Estimation_and_Inference/           # Tasks 3–4. IN PROGRESS.
│   ├── scripts/                                  # 03_estimation.py, 04_inference.py (to be written)
│   ├── outputs/                                  # Populated by Mohamed.
│   └── report/README.md                          # Brief describing what goes here.
│
├── Section_C_Model_Selection_and_Diagnostics/    # Tasks 5–6. IN PROGRESS.
│   ├── scripts/                                  # 05_model_selection.py, 06_diagnostics.py (to be written)
│   ├── outputs/                                  # Populated by Ayesha.
│   └── report/README.md                          # Brief describing what goes here.
│
├── requirements.txt
├── .gitignore
└── README.md                                     # This file.
```

## Section ownership

| Section | Tasks | Owner | Status |
|---------|-------|-------|--------|
| A — Preprocessing and Model Definition | 1, 2 | **<Ali>** | Done |
| B — Estimation and Inference           | 3, 4 | Mohamed               | In progress |
| C — Model Selection and Diagnostics    | 5, 6 | Ayesha                | In progress |

> **Note for the repo owner:** I did not know your name, so the Section A
> owner is a placeholder (`<YOUR NAME HERE>`). Please fill it in before
> sharing the repo with the team.

## Shared data — hands off from Sections B and C

The `data/` folder sits at the repo root and is shared across all
sections. Sections B and C should **read** from `data/` but should **not**
modify or regenerate the CSVs there. The datasets were produced by
`Section_A_.../scripts/01_preprocessing.py` and
`Section_A_.../scripts/02_model_definition.py` and are considered frozen
for the duration of the project. If you think a preprocessing choice is
wrong, raise it — don't patch the data files.

## Setup

```bash
# 1. Clone the repo.
git clone <repo-url>
cd crime-project

# 2. (Recommended) Create and activate a virtual environment.
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies.
pip install -r requirements.txt
```

## How to run Section A

From the repo root:

```bash
python Section_A_Preprocessing_and_Model_Definition/scripts/01_preprocessing.py
python Section_A_Preprocessing_and_Model_Definition/scripts/02_model_definition.py
```

Each script resolves its paths relative to this file's location, so it
will work regardless of the current working directory. Scripts in later
sections should follow the same convention.
