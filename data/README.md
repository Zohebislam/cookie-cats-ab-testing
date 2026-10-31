# Data

`raw/cookie_cats.csv`: 90,189 players of the mobile puzzle game Cookie Cats (Tactile Entertainment),
randomly assigned to have the first gate at level 30 or level 40.

The dataset comes from DataCamp's "Mobile Games A/B Testing with Cookie Cats" project and is also
published on Kaggle. The copy here was checked against two independent public copies; they are
identical.

| Column | Meaning |
|---|---|
| `userid` | Player ID |
| `version` | `gate_30` (control) or `gate_40` (variant) |
| `sum_gamerounds` | Game rounds played in the first 14 days after install |
| `retention_1` | Came back and played 1 day after installing |
| `retention_7` | Came back and played 7 days after installing |
