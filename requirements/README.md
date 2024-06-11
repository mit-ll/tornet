# Environment set up

## Conda

Replace {backend} with tensorflow, torch or jax.
```
conda create -n keras-{backend} python=3.10
conda activate keras-{backend}
pip install -r requirements/requirements-{backend}.txt
```

**Notes**:
- Tensorflow environment might not properly set up paths to cuda libraries, 
leading to the GPU not being registered. See requirements/fix_cuda_paths.sh for 
fix. 