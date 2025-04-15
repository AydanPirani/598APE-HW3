# 598APE-HW3

FOR REVIEWERS

Before running the code, please run the following commands after cloning the code (from the project directory):

chmod +x ./setup.sh
sudo sh setup.sh

To run a benchmark, checkout the relevant git commit and run:

hyperfine "<COMMAND>"

Note that you must replace <COMMAND> with the new command. Please ensure that the quotations around the command remain, as they are a hyperfine requirement.

Benchmarks (and Commit Hashes):
- Baseline: 4745d36519d2483cf67f2abe98ce980647056145
- Accumulator: f7446680d285b25a38f1ef2dbf372abe2b0dd89c
- AOS To SOA: c83e6102130ad0350278a4bb0dcdf68a028ab135
- OpenMP: 6f602e8552b6ba7864523ae71bb2cb56e1864f53
- Selective OpenMP: e7131045795292910a00dbadb1f32eb15ae432c4
- Vectorization (Final Accuracy-Preserving Commit): 760ae021f9676116d3eb41c8676f04638f6d32ac
- Rsqrt: 6f1e5a41042c19b1cb2d04ce3b868230aa0b4138


This repository contains code for homework 3 of 598APE.

This assignment is relatively simple in comparison to HW1 and HW2 to ensure you have enough time to work on the course project.

In particular, this repository is an implementation of an n-body simulator.

To compile the program run:
```bash
make -j
```

To clean existing build artifacts run:
```bash
make clean
```

This program assumes the following are installed on your machine:
* A working C compiler (g++ is assumed in the Makefile)
* make

The nbody program is a classic physics simulation whose exact results are unable to be solved for exactly through integration.

Here we implement a simple time evolution where each iteration advances the simulation one unit of time, according to Newton's law of gravitation.

Once compiled, one can call the nbody program as follows, where nplanets is the number of randomly generated planets for the simulation, and timesteps denotes how long to run the simulation for:
```bash
./main.exe <nplanets> <timesteps>
```

In particular, consider speeding up simple run like the following (which runs ~6 seconds on my local laptop under the default setup):
```bash
./main.exe 1000 5000
```

Exact bitwise reproducibility is not required, but approximate correctness (within a reasonable region of the final location).