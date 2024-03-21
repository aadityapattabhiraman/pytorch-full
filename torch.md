# Document Title

### Tensors

#### is_tensor
torch.is_tensor(obj)

Returns True if obj is a PyTorch Tensor. Uses isinstance(obj, Tensor). Hence it is recommended over is_tensor

"Parameters"
    obj(Object) -> object to test
    
(eg) torch.is_tensor(x)

#### is_storage
torch.is_storage(obj)

Returns True if obj is a PyTorch storage object.

"Parameters"
    obj(Object) -> Object ot test
    
#### is_complex
torch.is_complex(input)

Returns True if the data type of input is a complex data type (i.e) one of torch.complex64, torch.complex128

"Parameters"
    input(Tensor) -> the input tensor

#### is_conj
torch.is_conj(input)

Returns True if the input is a conjugated tensor (i.e) its conjugate bit is set to True

"Parameters"
    input(Tensor) -> the input tensor
    
#### is_floating_point
#### is_nonzero
#### set_default_dtype
torch.set_default_dtype(d)

Sets the default floating point dtype to d. Supports torch.float32 and torch.float64 as inputs. Other dtypes may be accepted without complaint but are not supported and are unlikely to work as expected.

"Parameters"
    d(torch.dtype) -> the floating point dtype to make the default
    
(eg) torch.set_default_dtype(torch.float64)

#### set_printoptions
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

"Parameters"
        precision – Number of digits of precision for floating point output (default = 4).
        threshold – Total number of array elements which trigger summarization rather than full repr (default = 1000).
        edgeitems – Number of array items in summary at beginning and end of each dimension (default = 3).
        linewidth – The number of characters per line for the purpose of inserting line breaks (default = 80). Thresholded matrices will ignore this parameter.
        profile – Sane defaults for pretty printing. Can override with any of the above options. (any one of default, short, full)
        sci_mode – Enable (True) or disable (False) scientific notation. If None (default) is specified, the value is defined by torch._tensor_str._Formatter. This value is automatically chosen by the framework.

#### tensor
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor

Constructs a tensor with no autograd history (also known as a “leaf tensor”, see Autograd mechanics) by copying data.


Parameters
    data (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
Keyword Arguments
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, infers data type from data.
        device (torch.device, optional) – the device of the constructed tensor. If None and data is a tensor then the device of data is used. If None and data is not a tensor then the result tensor is constructed on the current device.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.
        pin_memory (bool, optional) – If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: False.

#### asarray
torch.asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False) → Tensor

Converts obj to a tensor.

obj can be one of:
    a tensor
    a NumPy array or a NumPy scalar
    a DLPack capsule
    an object that implements Python’s buffer protocol
    a scalar
    a sequence of scalars


Parameters
    obj (object) – a tensor, NumPy array, DLPack Capsule, object that implements Python’s buffer protocol, scalar, or sequence of scalars.
Keyword Arguments
        dtype (torch.dtype, optional) – the datatype of the returned tensor. Default: None, which causes the datatype of the returned tensor to be inferred from obj.
        copy (bool, optional) – controls whether the returned tensor shares memory with obj. Default: None, which causes the returned tensor to share memory with obj whenever possible. If True then the returned tensor does not share its memory. If False then the returned tensor shares its memory with obj and an error is thrown if it cannot.
        device (torch.device, optional) – the device of the returned tensor. Default: None, which causes the device of obj to be used. Or, if obj is a Python sequence, the current default device will be used.
        requires_grad (bool, optional) – whether the returned tensor requires grad. Default: False, which causes the returned tensor not to require a gradient. If True, then the returned tensor will require a gradient, and if obj is also a tensor with an autograd history then the returned tensor will have the same history.

#### as_tensor 
torch.as_tensor(data, dtype=None, device=None) → Tensor

Converts data into a tensor, sharing data and preserving autograd history if possible.
If data is already a tensor with the requested dtype and device then data itself is returned, but if data is a tensor with a different dtype or device then it’s copied as if using data.to(dtype=dtype, device=device).
If data is a NumPy array (an ndarray) with the same dtype and device then a tensor is constructed using torch.from_numpy().


Parameters
        data (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, infers data type from data.
        device (torch.device, optional) – the device of the constructed tensor. If None and data is a tensor then the device of data is used. If None and data is not a tensor then the result tensor is constructed on the current device.

#### from_file
torch.from_file(filename, shared=None, size=0, *, dtype=None, layout=None, device=None, pin_memory=False)

Creates a CPU tensor with a storage backed by a memory-mapped file.
If shared is True, then memory is shared between processes. All changes are written to the file. If shared is False, then changes to the tensor do not affect the file.
size is the number of elements in the Tensor. If shared is False, then the file must contain at least size * sizeof(dtype) bytes. If shared is True the file will be created if needed.

Parameters
        filename (str) – file name to map
        shared (bool) – whether to share memory (whether MAP_SHARED or MAP_PRIVATE is passed to the underlying mmap(2) call)
        size (int) – number of elements in the tensor

Keyword Arguments
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()).
        layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        pin_memory (bool, optional) – If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: False.

#### from_numpy
torch.from_numpy(ndarray) → Tensor

Creates a Tensor from a numpy.ndarray.
The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. The returned tensor is not resizable.
It currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16, numpy.complex64, numpy.complex128, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, and bool.

#### zeros
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.


Parameters
    size (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
    
Keyword Arguments
        out (Tensor, optional) – the output tensor.
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()).
        layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

#### zeros_like
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

Returns a tensor filled with the scalar value 0, with the same size as input. torch.zeros_like(input) is equivalent to torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).


Parameters
    input (Tensor) – the size of input will determine size of the output tensor.
    
Keyword Arguments
        dtype (torch.dtype, optional) – the desired data type of returned Tensor. Default: if None, defaults to the dtype of input.
        layout (torch.layout, optional) – the desired layout of returned tensor. Default: if None, defaults to the layout of input.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, defaults to the device of input.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.
        memory_format (torch.memory_format, optional) – the desired memory format of returned Tensor. Default: torch.preserve_format.

#### arange
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

Returns a 1-D tensor of size ⌈end−start/step​⌉ with values from the interval [start, end) taken with common difference step beginning from start.
Note that non-integer step is subject to floating point rounding errors when comparing against end; to avoid inconsistency, we advise subtracting a small epsilon from end in such cases.


Parameters
        start (Number) – the starting value for the set of points. Default: 0.
        end (Number) – the ending value for the set of points
        step (Number) – the gap between each pair of adjacent points. Default: 1.

Keyword Arguments
        out (Tensor, optional) – the output tensor.
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()). If dtype is not given, infer the data type from the other input arguments. If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype(). Otherwise, the dtype is inferred to be torch.int64.
        layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

#### range
torch.range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

Returns a 1-D tensor of size ⌊end−start/step​⌋+1 with values from start to end with step step. Step is the gap between two values in the tensor.


Parameters
        start (float) – the starting value for the set of points. Default: 0.
        end (float) – the ending value for the set of points
        step (float) – the gap between each pair of adjacent points. Default: 1.

Keyword Arguments
        out (Tensor, optional) – the output tensor.
        dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()). If dtype is not given, infer the data type from the other input arguments. If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype(). Otherwise, the dtype is inferred to be torch.int64.
        layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

#### linspace
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive

Parameters
        start (float or Tensor) – the starting value for the set of points. If Tensor, it must be 0-dimensional
        end (float or Tensor) – the ending value for the set of points. If Tensor, it must be 0-dimensional
        steps (int) – size of the constructed tensor

Keyword Arguments
        out (Tensor, optional) – the output tensor.
        dtype (torch.dtype, optional) – the data type to perform the computation in. Default: if None, uses the global default dtype (see torch.get_default_dtype()) when both start and end are real, and corresponding complex dtype when either is complex.
        layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

