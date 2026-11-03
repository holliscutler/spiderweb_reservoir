| Task | Spider web | Default random | Web vs default | p |
|---|---|---|---|---|
| Mackey-Glass, 20 steps | 0.0023 (0.0022–0.0024) | 0.0034 (0.0031–0.0036) | 31% better | 2e-06 |
| Mackey-Glass, 100 steps | 0.1121 (0.1069–0.1168) | 0.1340 (0.1297–0.1387) | 16% better | 4e-06 |
| Santa Fe laser, 1 step | 0.0514 (0.0498–0.0532) | 0.0524 (0.0508–0.0541) | 2% better | 0.5 |
| Memory capacity | 36.8 (35.7–37.9) | 45.8 (44.7–46.9) | 20% worse | 1e-07 |

| Hardware (mean of 20) | Spider web | Default random | Web / default |
|---|---|---|---|
| Operations per time step | 4,261 | 4,262 | 1.00 |
| Total wire, web's layout | 17,111 | 31,508 | 0.54 |
| Total wire, best placement | 17,106 | 25,756 | 0.66 |
| Input wire | 10 | 102 | 0.10 |

| Tuned settings | Mackey-Glass, 20 steps | Mackey-Glass, 100 steps | Santa Fe laser, 1 step | Memory capacity |
|---|---|---|---|---|
| Spider web (sr, lr, input scaling, ridge) | 1, 0.3, 1, 1e-08 | 1, 0.3, 1, 1e-08 | 0.9, 0.6, 0.5, 1e-08 | 1, 1, 0.1, 0.0001 |
| Default random (sr, lr, input scaling, ridge) | 1, 0.6, 1, 1e-08 | 1, 0.3, 1, 1e-08 | 1, 0.6, 0.5, 1e-08 | 1, 1, 0.1, 0.0001 |
