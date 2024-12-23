# Astronav

Long-range route-plotting tool for Elite Dangerous.

## Features

- Advanced graph search algorithms for route computation:
  - High performance beam-search algorithm for finding the shortest (in terms of jumps) path between two systems.
  - - Custom heuristic for the beam-search algorithm that tries to minimize the number of jumps.
  - A-Star and Dijkstra algorithms for finding the most fuel efficient route between systems.
- Memory-mapped, precomputed KD-Tree loads almost instantly and allows for fast nearest-neighbor searches and low memory usage.
- Typo-resistant, high-performance system name search.
- Fuel-consumption is taken into account when plotting routes.
- User-friendly command-line interface with built-in help.
- Pretty colors in output.
- (very WIP) GUI for plotting routes.
- Python API for building your own GUI or web interface and integrating Astronav into your own tools.

## Performance

Astronav is designed to be fast. It uses a beam-search algorithm to find the shortest path between two systems. This algorithm is usually on-par with a full breadth-first search, but is much faster. For example, plotting a route from Sol to Colonia (a distance of 22,000 light-years) takes about 8 seconds on my machine with a beam-width of 8192 and results in a route of 130 Ly. The same route takes about 2 minutes with a full breadth-first search. Peak memory usage is around 400 MB for the beam-search and around 2 GB for the full breadth-first search with approximately 140 million star system positions loaded. The routes are 129 jumps and 122 jumps respectively. In conclusion, the beam-search algorithm is about 30-50x faster than a full breadth-first search while giving a result of comparable quality.