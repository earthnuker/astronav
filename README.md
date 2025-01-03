# Astronav

[![asciicast](https://asciinema.org/a/qKKhN6AyMHfQr1VZNHpTUNQpk.svg)](https://asciinema.org/a/qKKhN6AyMHfQr1VZNHpTUNQpk)

Long-range route-plotting tool for Elite Dangerous.

## Features

- Advanced graph search algorithms for route computation:
  - High performance beam-search algorithm for finding the shortest (in terms of jumps) path between two systems.
  - Custom heuristic for the beam-search algorithm that tries to minimize the number of jumps.
  - A-Star and Dijkstra algorithms for finding the most fuel efficient route between systems.
- Memory-mapped, precomputed KD-Tree loads almost instantly and allows for fast nearest-neighbor searches and low memory usage.
- Typo-resistant, high-performance, fuzzy system name search using Levenshtein distance.
- Fuel-consumption and refuel stops are taken into account when plotting routes.
- User-friendly command-line interface with built-in help.
- Pretty colors in output.
- (very WIP) GUI for plotting routes.
- Python API for building your own GUI or web interface and integrating Astronav into your own tools.

## Running Astronav

1. Install astronav using cargo:
```bash
cargo install --git https://github.com/earthnuker/astronav.git
```

3. Download the latest galaxy dump from Spansh  [here](https://downloads.spansh.co.uk/galaxy.json.gz), this is around 80GB compressed, so make sure you have enough disk space. (Eventually, I will provide a smaller, preprocessed version of the galaxy dump that only contains the data Astronav needs.)
4. Preprocess the galaxy dump by running the following command:
```bash
astronav preprocess-galaxy X:\path\to\out\folder Y:\path\to\galaxy.json.gz
```

1. Run Astronav with the following command:

```bash
astronav route --stars-path X:\path\to\out\folder --ship NMGR --mmap beam --beam-width 8192 --refuel-mode least-jumps Sol Colonia
```

## Performance

Astronav is designed to be fast. It uses a beam-search algorithm to find the shortest path between two systems. This algorithm is usually on-par with a full breadth-first search, but is much faster. For example, plotting a route from Sol to Colonia (a distance of 22,000 light-years) takes about 8 seconds on my machine with a beam-width of 8192 and results in a route of 130 Ly. The same route takes about 2 minutes with a full breadth-first search. Peak memory usage is around 400 MB for the beam-search and around 2 GB for the full breadth-first search with approximately 140 million star-system positions loaded. The routes are 129 jumps and 122 jumps respectively. In conclusion, the beam-search algorithm is about 30-50x faster than a full breadth-first search while giving a result of comparable quality.

On average, when comparing a beam-search with a width of 1024 and a full breadth-first search, the beam-search is about 160-170x faster while returning routes within 20 jumps of the optimal route and a beam-width of 8192 is about 60x faster and returns results within 5-10 jumps of the optimal route.

The following table shows average results from computing around 100 routes between random pairs of systems in the galaxy with different beam widths.
These computations were run utilizing 8 threads on a Ryzen 9 5950X clocked at 4 GHz with 32 GB of RAM and KD-Tree memory-mapped in from an NVMe SSD (WD_BLACK SN770).


| Beam Width     | Mean Speedup  | Mean Length difference from optimal route (Jumps) |
| -------------- | ------------- | ------------------------------------------------- |
| 64             | 353x          | 33.59                                             |
| 128            | 356x          | 27.88                                             |
| 256            | 303x          | 24.99                                             |
| 512            | 252x          | 17.88                                             |
| 1024           | 200x          | 13.37                                             |
| 2048           | 145x          | 10.86                                             |
| 4096           | 98x           | 8.57                                              |
| 8192           | 61x           | 6.05                                              |
| $\infty$ (BFS) | 1x (baseline) | 0.00                                              |

The heuristic used to select nodes in the graph to expand next uses the following formula to rank nodes:

$$
\text{max}(\text{dist}(node, goal) - \text{mult}(node)  \cdot range, 0)
$$

Where $\text{dist}(node, goal)$ is the distance between the node and the goal system in light-years, $\text{mult}(node)$ is a multiplier that depends on the star type of the node (1 for normal stars, 1.5 for white dwarf stars and 4 for neutron stars) and $range$ is the jump range of the ship.
This computes the remaining distance to the goal system (in light-years) if we were to jump from here in a straight line towards it.