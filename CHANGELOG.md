## 2.0.4 (2025-12-30)

### Fix

- pin jax and mpi4jax and remove tokens

## 2.0.3 (2025-12-24)

### Fix

- import solutions submodule in init

## 2.0.2 (2025-11-26)

### Fix

- fixes to pytree construction

## 2.0.1 (2025-06-02)

### Fix

- change typehint for input noise class

### Perf

- switch to jax fori_loop

## 2.0.0 (2025-02-08)

### Feat

- add mpi support to solutions, and ivar to maps, several doc fixes, rename to avoid namespace collision

## 1.0.0 (2025-01-24)

### Feat

- rename solution params to data, add function to get map hits, add map noise models

### Refactor

- make solutions a submodule and break map class out to its own file

## 0.9.1 (2025-01-06)

### Fix

- copy comm in todvec

## 0.9.0 (2024-11-12)

### Feat

- add module for useful math functions

### Fix

- correct DCT usage in NoiseSmoothedSVD

## 0.8.0 (2024-11-05)

### Feat

- add mpi support to todvec
- add indentity noise and helper function for recomputing noise

### Fix

- compute lims properly, better error handling on noise computation, and make sure arrays are jax arrays
- dont cast in pure callback

## 0.7.0 (2024-10-11)

### Feat

- add support for external noise classes

## 0.6.0 (2024-10-08)

### Feat

- more flexability in noise recomputation

## 0.5.1 (2024-08-09)

### Fix

- speed up and clean up math functions, fixed some typing issues
- use dataclass default factory and add missing import

## 0.5.0 (2024-07-30)

### Feat

- add pcg solver
- expanded math operators
- add matmul to solutions
- add nsamp property

### Refactor

- move lhs and rhs to mapmaking module

## 0.4.0 (2024-07-12)

### Feat

- add function to generate empty map
- add functions to compute lhs and rhs of the mapmaker eq

## 0.3.0 (2024-07-11)

### Feat

- add framework for solution terms and basic map solution

### Fix

- fix TODVec iterator and clearer docs

## 0.2.0 (2024-07-10)

### Feat

- add basic noise models

## 0.1.0 (2024-07-07)

### Feat

- added basic implementation of TOD class
