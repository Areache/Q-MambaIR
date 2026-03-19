import math
# import fast_hadamard_transform_cuda
from fast_hadamard_transform import hadamard_transform #HadamardTransformFn, HadamardTransform12NFn, HadamardTransform40NFn 

# import quant_hadamard

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def get_had_fn(dim):
    had_scale = 1.0 / math.sqrt(dim) # hadamard transform scaling factor
    if dim % 12 == 0:
        N = 12
        if (is_pow2(dim // 12)):
            transform_fn = hadamard_transform # HadamardTransform12NFn
        else:
            transform_fn = hadamard_transform # HadamardTransformFn
        # transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_12N
        # transform_fn = fast_hadamard_transform
    elif dim % 40 == 0:
        N = 40
        if (is_pow2(dim // 40)):
            transform_fn = hadamard_transform # HadamardTransform40NFn 
        else:
            transform_fn = hadamard_transform # HadamardTransformFn      
        # transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_40N
        # transform_fn = fast_hadamard_transform
    else:
        N = 2
        if (is_pow2(dim)):
           transform_fn = hadamard_transform # HadamardTransformFn 
        # transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform
        # transform_fn = fast_hadamard_transform
    return transform_fn, N, had_scale


def get_qhad_fn(dim):
    had_scale = 1.0 / math.sqrt(dim) # hadamard transform scaling factor
    if dim % 12 == 0:
        assert (is_pow2(dim // 12))
        N = 12
        # transform_fn = quant_hadamard.fast_hadamard_transform_12N
        transform_fn = fast_hadamard_transform
    elif dim % 40 == 0:
        assert (is_pow2(dim // 40))
        N = 40
        # transform_fn = quant_hadamard.fast_hadamard_transform_40N
        transform_fn = fast_hadamard_transform
    else:
        assert (is_pow2(dim))
        N = 2
        # transform_fn = quant_hadamard.fast_hadamard_transform
        transform_fn = fast_hadamard_transform
    return transform_fn, N, had_scale