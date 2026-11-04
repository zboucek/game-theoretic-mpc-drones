# Reference dataset

The complete numerical evidence for this software release is identified by:

- **Dataset DOI:** [10.5281/zenodo.23947592](https://doi.org/10.5281/zenodo.23947592)
- **Dataset version:** 1.0.0
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Software:** [game-theoretic-mpc-drones](https://github.com/zboucek/game-theoretic-mpc-drones)

Download `release-game-theory-mpc-article-data-1.0.0.tar.xz` from the dataset record's **Files** section.
If the DOI has only been reserved, the files become publicly available when the
Zenodo record is published. Keep the archive compressed and run from the software
repository:

```bash
python -m game_theory_mpc verify --audit-data /path/to/release-game-theory-mpc-article-data-1.0.0.tar.xz
```

The verifier checks the archive and record checksums and performs the complete
independent numerical audit. Figure/table regeneration and custom experiments
work without the full dataset.

## Archive integrity

- **SHA-256:** `d6d94ad9ceac6219bcbfbe36208d3ef87ea6069e89325803b2d443d78e8a92a7`
- **Size:** 69,228,368 bytes
- **Retained data files:** 16,319

Cite the dataset using its DOI and cite the accompanying article when discussing
the methods or scientific results. The dataset DOI identifies the data; the
article citation is recorded separately in the software's `CITATION.cff`.
