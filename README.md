# Performance of HPCC amd20 nodes 

See the [ICER Wiki](https://wiki.hpcc.msu.edu/display/ITH/Cluster+amd20+with+AMD+CPUs) for detailed info on the nodes.
In all tests, performance metrics should be computed using only the `evolution` timer to avoid the `initialization` cost. 
Useful performance metric is zone-updates per core-second (ZUPCS). 

## Test problems

1. 3D Sod. This is a great test because the number of blocks is completely controllable (and more or less predictable). Test should be setup with about 5 levels of refinment. Grid-aligned shock is fine. The number of blocks in the simulation is then easily controllable by changing `nblock{xyz}` at runtime. Setup line:

```bash
> ./setup  Sod -auto -3d -nxb=12 -nyb=12 -nzb=12 -objdir obj_sod3d +spark threadWithinBlock=True -maxblocks=200
```

## Single-node tests

Start by exploring the strong scaling and single-node performance. The number of OpenMP threads should be varied. 

1. Pure MPI, `OMP_NUM_THREADS=1`. 
  - Strong scaling, total `leaf_blocks` = {128,256,512}. Vary number of MPI ranks from 1 to 128 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 128 in powers of two.
Metric should be ZUPCS. 

2. `OMP_NUM_THREADS=2`.
  - Strong scaling, total `leaf_blocks` = {64,128,256}. Vary number of MPI ranks from 1 to 64 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 64 in powers of two.

3. `OMP_NUM_THREADS=4`.
  - Strong scaling, total `leaf_blocks` = {32,64,128}. Vary number of MPI ranks from 1 to 32 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 32 in powers of two.
  
4. `OMP_NUM_THREADS=8`.
  - Strong scaling, total `leaf_blocks` = {16,32,64}. Vary number of MPI ranks from 1 to 16 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 16 in powers of two.

5. `OMP_NUM_THREADS=16`.
  - Strong scaling, total `leaf_blocks` = {8,16,32}. Vary number of MPI ranks from 1 to 8 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 8 in powers of two.

6. `OMP_NUM_THREADS=32`.
  - Strong scaling, total `leaf_blocks` = {4,8,16}. Vary number of MPI ranks from 1 to 4 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 4 in powers of two.

7. `OMP_NUM_THREADS=64`.
  - Strong scaling, total `leaf_blocks` = {2,4,8}. Vary number of MPI ranks from 1 to 2 in powers of two.
  - Weak scaling, `leaf_blocks` per rank = {1,2,4}. Vary number of MPI ranks from 1 to 2 in powers of two.
