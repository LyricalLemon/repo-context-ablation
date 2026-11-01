# Where does repository context help?

COM3001 final year project, BSc Computer Science, University of Surrey (2026 to 2027). Supervisor: Dany Varghese.

This repository holds the code for a controlled ablation of retrieval for code language models. It tests which design choices in repository-level retrieval improve code completion by small open-weight models, and measures which generated tokens the retrieved context actually helps.

## Status

Early setup. Nothing here is a result yet.

## Layout

| Folder | Contents |
| --- | --- |
| `src/` | Pipeline code: chunkers, retrievers, symbol graph, prompt assembly, measurement |
| `tests/` | Unit tests for each component |
| `configs/` | One configuration file per experiment run |
| `scripts/` | Entry points that run experiments and rebuild tables and figures |
| `results/` | Small summary tables only |

Model weights, cloned repositories, datasets and raw logs are not committed. See `.gitignore`.

## Setup

To be written once the environment is fixed. Every run will record its configuration, model name and precision, seed and git commit.

## Use of AI

AI-generated code is identified in the commit message and in the source. See the AI-use commentary in the report.

## Licence

None yet. All rights reserved until the University IP position is confirmed.
