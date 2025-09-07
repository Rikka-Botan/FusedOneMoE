# FusedOneMoE

Faster Mixture of Experts by using matmul (matrix multiplication)

<img width="4400" height="2475" alt="FusedOneMoE" src="https://github.com/user-attachments/assets/eba53256-9ed1-47c6-b82b-f22b85262ae6" />

## Features

1. Efficient operations by using matrix multiplication
   General GPUs are designed to perform matrix multiplication faster than other operations.

2. Static operational graph by avoiding “for” and “if”
   Dynamic operational graph is difficult to optimize kernel operations.

3. Optimizations within the library is available
   Machine learning libraries, such as PyTorch, are highly optimized within itself.

***

## How to use

```python
from modeling import FusedOneMoE

model = FusedOneMoE(
)


```
