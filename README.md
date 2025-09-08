# FusedOneMoE

Faster Kernel Fused Mixture of Experts by using matmul (matrix multiplication)

(Dense MoE fwd: only one line implementation!)

<img width="4400" height="2475" alt="FusedOneMoE" src="https://github.com/user-attachments/assets/eba53256-9ed1-47c6-b82b-f22b85262ae6" />

## Features

### 1. Efficient operations by using matrix multiplication
   General GPUs are designed to perform matrix multiplication faster than other operations.

### 2. Static operational graph by avoiding “for” and “if”
   Dynamic operational graph is difficult to optimize kernel operations.

### 3. Optimizations within the library is available
   Machine learning libraries, such as PyTorch, are highly optimized within itself.

***

# Speed Comparison to usual MoE implements

## hidden size: 128, intermediate size: 512, sequence lenght: 512, experts num: 32, activate num: 8

|Module             |Time(Intel Ultra 7 265K)  |
|:--                |:--                       |
|usual MoE          |2.660 sec                 |
|Fused One MoE      |0.313 sec                 |

### Accelerataion rate: 8.50x


## hidden size: 128, intermediate size: 512, sequence lenght: 512, experts num: 128, activate num: 8

|Module             |Time(Intel Ultra 7 265K)  |
|:--                |:--                       |
|usual MoE          |7.468 sec                 |
|Fused One MoE      |1.142 sec                 |

### Accelerataion rate: 6.54x

## hidden size: 128, intermediate size: 512, sequence lenght: 512, experts num: 1024, activate num: 8

|Module             |Time(Intel Ultra 7 265K)  |
|:--                |:--                       |
|usual MoE          |53.797 sec                |
|Fused One MoE      |8.853 sec                 |

### Accelerataion rate: 6.08x


## How to use

```python
"""
Args:
   hidden_size: int
   intermediate_size: int
   groups: int (experts num)
   is_sparse: bool (dense or sparse MoE)
   topk: int
"""

from modeling import FusedOneMoE.FusedOneMoE

model = FusedOneMoE(
   hidden_size = 128, intermediate_size = 512, groups = 32, is_sparse = True, topk = 8
)

y = model(x)
```

## Acknowledgements

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

## Citations

I would be happy to include a citation at the end, but it is not required.

Feel free to use this model.


## Contact Us

[My X account](https://x.com/peony__snow)


## About Author

### Rikka Botan

Japanese independent researcher having shy and pampered personality >_<

Twin-tail hair is a charm point :)

Interested in natural language processings. 

Usually using python and C.

<img width="4405" height="2480" alt="RikkaBotan_Logo" src="https://github.com/user-attachments/assets/71ed04a3-dce0-4253-b91e-a0e16e7812f5" />

