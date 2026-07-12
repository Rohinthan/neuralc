"""
neuralc.py — Python bindings for the neuralc C library via ctypes.

This file is built directly against the actual C sources (tensor.c/.h,
conv.c/.h, layer.c/.h, cuda_backend.h) rather than assumptions about their
behavior — see the inline notes marked "per tensor.c" / "per conv.c" etc.
for the specific ground-truth detail each design choice is based on.

Usage:
    from neuralc import Tensor, Network, Dense, Conv2D, MaxPool2D, Adam, ACT_RELU, ACT_SIGMOID

Requirements:
    Build the shared library first (from the project root, not python/):
        make libneuralc

IMPORTANT — fail-fast C error handling:
    neuralc's C library uses cforge_error() for invalid states, which calls
    exit(1) and terminates the ENTIRE PROCESS — it is not a raised C++/Python
    exception and cannot be caught. This is most likely to bite you around
    GPU device transfers (calling .to('cuda') on a CPU-only build, migrating
    a view tensor, etc.). Every method in this file that could otherwise
    trigger a cforge_error() call is guarded with an upfront Python-side
    check (cf_cuda_enabled(), cf_opencl_enabled(), tensor ownership) that
    raises a normal, catchable Python exception instead. If you call raw
    _lib.* functions directly rather than going through these wrappers, you
    lose those guards.

Example (MLP):
    import neuralc as nc

    x = nc.Tensor.zeros([4, 2])
    y = nc.Tensor.zeros([4, 1])

    net = nc.Network()
    net.add(nc.Dense(2, 8,  nc.ACT_RELU))
    net.add(nc.Dense(8, 1,  nc.ACT_SIGMOID))
    net.set_output_dim(1)

    opt = nc.Adam(lr=0.01)
    for epoch in range(3000):
        loss = net.train_step(x, y, nc.LOSS_BINARY_CROSS)
        opt.step(net)

    pred = net.forward(x)

Example (CNN building blocks, CPU):
    conv = nc.Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, pad=1)
    pool = nc.MaxPool2D(pool_size=2, stride=2)

    x    = nc.Tensor.zeros([16, 1, 28, 28])   # (batch, C, H, W)
    fmap = conv.forward(x)                    # -> (16, 8, 28, 28)
    pooled = pool.forward(fmap)               # -> (16, 8, 14, 14)
    flat = pooled.flatten()                   # -> (16, 8*14*14), zero-copy view

Example (GPU — Conv2D/MaxPool2D are CUDA-only, no OpenCL kernels exist yet):
    if nc.cf_cuda_enabled():
        conv.to('cuda')
        x = nc.Tensor.zeros([16, 1, 28, 28]).to('cuda')
        fmap = conv.forward(x)   # forward()/backward() keep outputs on the
                                  # same device as the input automatically
"""

import ctypes
import os
import numpy as np

# ── load shared library ────────────────────────────────────────────────

def _find_lib():
    """Search for libneuralc.so in common locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "libneuralc.so"),
        os.path.join(os.path.dirname(__file__), "libneuralc.so"),
        "./libneuralc.so",
        "/usr/local/lib/libneuralc.so",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "libneuralc.so not found. Run 'make libneuralc' from the project "
        "root first.\nSearched: " + str(candidates)
    )

_lib = ctypes.CDLL(_find_lib())

# ── device / backend enums (per tensor.h: CF_Device, CF_GpuBackend) ───

CF_DEVICE_CPU = 0
CF_DEVICE_GPU = 1

CF_GPU_NONE   = 0
CF_GPU_CUDA   = 1
CF_GPU_OPENCL = 2

# ── C struct mirror (per tensor.h Tensor struct, field-for-field) ─────

TENSOR_MAX_DIMS = 8

class _CTensor(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.POINTER(ctypes.c_float)),
        ("shape",       ctypes.c_int * TENSOR_MAX_DIMS),
        ("ndim",        ctypes.c_int),
        ("size",        ctypes.c_size_t),
        ("owns_data",   ctypes.c_int),
        ("device",      ctypes.c_int),   # CF_Device
        ("gpu_backend", ctypes.c_int),   # CF_GpuBackend
    ]

# ── activation / loss enums ────────────────────────────────────────────
# NOTE: matches layer.h's Activation enum exactly. LOSS_* values are not
# defined in any header provided alongside this file (nn.h was not
# available) — they're carried over unchanged from the previously-working
# build and were validated empirically via a converging XOR training run,
# but are not independently confirmed against nn.c source.

ACT_NONE    = 0
ACT_RELU    = 1
ACT_SIGMOID = 2
ACT_TANH    = 3
ACT_SOFTMAX = 4

LOSS_MSE           = 0
LOSS_BINARY_CROSS  = 1
LOSS_CROSS_ENTROPY = 2

# ── C function signatures ──────────────────────────────────────────────

def _setup_signatures():
    P = ctypes.POINTER

    # ── tensor lifecycle (tensor.h) ──
    _lib.tensor_create.restype  = P(_CTensor)
    _lib.tensor_create.argtypes = [P(ctypes.c_int), ctypes.c_int]
    _lib.tensor_zeros.restype   = P(_CTensor)
    _lib.tensor_zeros.argtypes  = [P(ctypes.c_int), ctypes.c_int]
    _lib.tensor_free.restype    = None
    _lib.tensor_free.argtypes   = [P(_CTensor)]
    _lib.tensor_clone.restype   = P(_CTensor)
    _lib.tensor_clone.argtypes  = [P(_CTensor)]

    # ── tensor ops (tensor.h) ──
    _lib.tensor_add.restype     = None
    _lib.tensor_add.argtypes    = [P(_CTensor), P(_CTensor), P(_CTensor)]
    _lib.tensor_matmul.restype  = None
    _lib.tensor_matmul.argtypes = [P(_CTensor), P(_CTensor), P(_CTensor)]
    _lib.tensor_sum.restype     = ctypes.c_float
    _lib.tensor_sum.argtypes    = [P(_CTensor)]
    _lib.tensor_mean.restype    = ctypes.c_float
    _lib.tensor_mean.argtypes   = [P(_CTensor)]
    _lib.tensor_argmax.restype  = ctypes.c_int
    _lib.tensor_argmax.argtypes = [P(_CTensor)]
    _lib.tensor_fill.restype    = None
    _lib.tensor_fill.argtypes   = [P(_CTensor), ctypes.c_float]
    _lib.tensor_copy_data.restype  = None
    _lib.tensor_copy_data.argtypes = [P(_CTensor), P(_CTensor)]
    _lib.tensor_shape_equal.restype  = ctypes.c_int
    _lib.tensor_shape_equal.argtypes = [P(_CTensor), P(_CTensor)]

    # tensor_reshape (used internally by conv.c's flatten(), exposed here
    # too since it's a generally useful zero-copy view operation)
    _lib.tensor_reshape.restype  = P(_CTensor)
    _lib.tensor_reshape.argtypes = [P(_CTensor), P(ctypes.c_int), ctypes.c_int]

    # ── GPU residency (tensor.h — confirmed against tensor.c) ──
    # tensor_to_gpu/_ex/tensor_to_cpu all return Tensor* (NOT void), and
    # all three internally call cforge_error() -> exit(1) on failure paths
    # (unsupported backend, no device, or migrating a non-owning view) —
    # see the Tensor.to() Python wrapper below for the guards that avoid
    # ever hitting those paths.
    _lib.tensor_to_gpu.restype     = P(_CTensor)
    _lib.tensor_to_gpu.argtypes    = [P(_CTensor)]
    _lib.tensor_to_gpu_ex.restype  = P(_CTensor)
    _lib.tensor_to_gpu_ex.argtypes = [P(_CTensor), ctypes.c_int]
    _lib.tensor_to_cpu.restype     = P(_CTensor)
    _lib.tensor_to_cpu.argtypes    = [P(_CTensor)]

    # cf_cuda_enabled/cf_opencl_enabled are always present regardless of
    # build flags (tensor.c defines them unconditionally, just returning 0
    # when the corresponding backend isn't compiled in) — safe to call
    # without any hasattr() guard, unlike conv2d_to_gpu/tensor_to_gpu etc.
    _lib.cf_cuda_enabled.restype   = ctypes.c_int
    _lib.cf_cuda_enabled.argtypes  = []
    _lib.cf_opencl_enabled.restype = ctypes.c_int
    _lib.cf_opencl_enabled.argtypes = []

    # ── nn / Network (nn.h source not available; signatures carried
    # over unchanged from a version validated by a converging XOR
    # training run — not independently re-verified against nn.c) ──
    _lib.nn_create.restype      = ctypes.c_void_p
    _lib.nn_create.argtypes     = []
    _lib.nn_add_layer.restype   = None
    _lib.nn_add_layer.argtypes  = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.nn_free.restype        = None
    _lib.nn_free.argtypes       = [ctypes.c_void_p]
    _lib.nn_forward.restype     = None
    _lib.nn_forward.argtypes    = [ctypes.c_void_p, P(_CTensor), P(_CTensor)]
    _lib.nn_loss.restype        = ctypes.c_float
    _lib.nn_loss.argtypes       = [ctypes.c_int, P(_CTensor), P(_CTensor), P(_CTensor)]
    _lib.nn_backward.restype    = None
    _lib.nn_backward.argtypes   = [ctypes.c_void_p, P(_CTensor)]
    _lib.nn_train_step.restype  = ctypes.c_float
    _lib.nn_train_step.argtypes = [ctypes.c_void_p, P(_CTensor), P(_CTensor),
                                   ctypes.c_int, P(_CTensor), P(_CTensor)]
    _lib.nn_save.restype        = ctypes.c_int
    _lib.nn_save.argtypes       = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.nn_load.restype        = ctypes.c_int
    _lib.nn_load.argtypes       = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.nn_accuracy_binary.restype      = ctypes.c_float
    _lib.nn_accuracy_binary.argtypes     = [P(_CTensor), P(_CTensor)]
    _lib.nn_accuracy_multiclass.restype  = ctypes.c_float
    _lib.nn_accuracy_multiclass.argtypes = [P(_CTensor), P(_CTensor)]

    # ── dense layer (layer.h, confirmed against layer.c) ──
    _lib.dense_create.restype   = ctypes.c_void_p
    _lib.dense_create.argtypes  = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.dense_param_count.restype  = ctypes.c_int
    _lib.dense_param_count.argtypes = [ctypes.c_void_p]
    # dense_to_gpu/_ex/dense_to_cpu ALWAYS exist as symbols and call
    # cforge_error() -> exit(1) when neither USE_CUDA nor USE_OPENCL is
    # compiled in (per layer.c) — guarded in Dense.to() below.
    _lib.dense_to_gpu.restype    = None
    _lib.dense_to_gpu.argtypes   = [ctypes.c_void_p]
    _lib.dense_to_gpu_ex.restype  = None
    _lib.dense_to_gpu_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.dense_to_cpu.restype    = None
    _lib.dense_to_cpu.argtypes   = [ctypes.c_void_p]

    # ── conv2d layer (conv.h, confirmed against conv.c) ──
    _lib.conv2d_create.restype   = ctypes.c_void_p
    _lib.conv2d_create.argtypes  = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
    _lib.conv2d_forward.restype  = None
    _lib.conv2d_forward.argtypes = [ctypes.c_void_p, P(_CTensor), P(_CTensor)]
    _lib.conv2d_backward.restype  = None
    _lib.conv2d_backward.argtypes = [ctypes.c_void_p, P(_CTensor), P(_CTensor)]
    _lib.conv2d_param_count.restype  = ctypes.c_int
    _lib.conv2d_param_count.argtypes = [ctypes.c_void_p]
    _lib.conv2d_free.restype  = None
    _lib.conv2d_free.argtypes = [ctypes.c_void_p]
    # conv2d_to_gpu/conv2d_to_cpu ALWAYS exist as symbols (per conv.c,
    # #ifndef USE_CUDA branches to cforge_error() rather than being
    # #ifdef'd out) and are CUDA-only — no OpenCL kernels exist for
    # Conv2D/MaxPool2D yet. Guarded via cf_cuda_enabled() in Conv2D.to().
    _lib.conv2d_to_gpu.restype  = None
    _lib.conv2d_to_gpu.argtypes = [ctypes.c_void_p]
    _lib.conv2d_to_cpu.restype  = None
    _lib.conv2d_to_cpu.argtypes = [ctypes.c_void_p]

    # ── pooling ops (conv.h, confirmed against conv.c) ──
    _lib.maxpool2d_forward.restype  = None
    _lib.maxpool2d_forward.argtypes = [P(_CTensor), P(_CTensor), P(_CTensor),
                                       ctypes.c_int, ctypes.c_int]
    _lib.maxpool2d_backward.restype  = None
    _lib.maxpool2d_backward.argtypes = [P(_CTensor), P(_CTensor), P(_CTensor),
                                        ctypes.c_int, ctypes.c_int]

    # ── flatten (conv.h — a view via tensor_reshape, owns_data=0) ──
    _lib.flatten.restype  = P(_CTensor)
    _lib.flatten.argtypes = [P(_CTensor)]

    # ── optimizers (optimizer.h source not available; carried over
    # unchanged from a version validated by a converging XOR training
    # run — not independently re-verified against optimizer.c) ──
    _lib.adam_create.restype    = ctypes.c_void_p
    _lib.adam_create.argtypes   = [ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _lib.adam_step.restype      = None
    _lib.adam_step.argtypes     = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.adam_free.restype      = None
    _lib.adam_free.argtypes     = [ctypes.c_void_p]

    _lib.sgd_create.restype     = ctypes.c_void_p
    _lib.sgd_create.argtypes    = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _lib.sgd_step.restype       = None
    _lib.sgd_step.argtypes      = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.sgd_free.restype       = None
    _lib.sgd_free.argtypes      = [ctypes.c_void_p]

    _lib.rmsprop_create.restype     = ctypes.c_void_p
    _lib.rmsprop_create.argtypes    = [ctypes.c_float, ctypes.c_float,
                                       ctypes.c_float, ctypes.c_float]
    _lib.rmsprop_step.restype       = None
    _lib.rmsprop_step.argtypes      = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.rmsprop_free.restype       = None
    _lib.rmsprop_free.argtypes      = [ctypes.c_void_p]

_setup_signatures()

# ── module-level GPU capability checks (tensor.c: always defined) ──────

def cf_cuda_enabled() -> bool:
    """True if libneuralc.so was built with -DUSE_CUDA AND a CUDA device
    is available at runtime. Always safe to call."""
    return bool(_lib.cf_cuda_enabled())

def cf_opencl_enabled() -> bool:
    """True if libneuralc.so was built with -DUSE_OPENCL AND an OpenCL
    device is available at runtime. Always safe to call."""
    return bool(_lib.cf_opencl_enabled())

# ── helpers ──────────────────────────────────────────────────────────

def _shape_arr(shape):
    return (ctypes.c_int * len(shape))(*shape)

def _out_dim(in_dim, kernel_size, stride, pad):
    """Standard convolution/pooling output-size formula.
    Matches conv2d_output_size() in conv.c exactly:
        out = (in_dim + 2*pad - kernel_size) / stride + 1
    """
    return (in_dim + 2 * pad - kernel_size) // stride + 1

# ── Tensor class ───────────────────────────────────────────────────────

class Tensor:
    """Python wrapper around neuralc's C Tensor struct (tensor.h)."""

    def __init__(self, ptr, owned=True):
        self._ptr   = ptr
        self._owned = owned
        self._base  = None  # set on views (e.g. flatten()) to keep the
                             # underlying buffer's owner alive

    def __del__(self):
        # Safe to call unconditionally when owned: tensor_free() (see
        # tensor.c) internally checks owns_data itself before freeing
        # `data`, and always frees the Tensor struct itself either way.
        # So even for a flatten()-produced view (owns_data=0), calling
        # tensor_free() here correctly frees just the small wrapper
        # struct, not the shared data buffer.
        if self._owned and self._ptr:
            _lib.tensor_free(self._ptr)
            self._ptr = None

    # ── factory methods ──

    @staticmethod
    def zeros(shape):
        arr = _shape_arr(shape)
        ptr = _lib.tensor_zeros(arr, len(shape))
        if not ptr: raise MemoryError("tensor_zeros failed")
        return Tensor(ptr)

    @staticmethod
    def from_numpy(array: np.ndarray):
        """Create a neuralc Tensor from a numpy float32 array."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        shape = list(array.shape)
        t     = Tensor.zeros(shape)
        src = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes.memmove(t._ptr.contents.data, src,
                       array.size * ctypes.sizeof(ctypes.c_float))
        return t

    def to_numpy(self) -> np.ndarray:
        """Convert to a numpy float32 array. Tensor must be CPU-resident."""
        if self.device == "gpu":
            raise RuntimeError(
                "to_numpy() requires a CPU-resident tensor — call "
                ".to('cpu') first."
            )
        c  = self._ptr.contents
        sz = c.size
        buf = (ctypes.c_float * sz).from_address(
                ctypes.addressof(c.data.contents))
        arr = np.frombuffer(buf, dtype=np.float32).copy()
        shape = [c.shape[i] for i in range(c.ndim)]
        return arr.reshape(shape)

    # ── properties ──

    @property
    def shape(self):
        c = self._ptr.contents
        return tuple(c.shape[i] for i in range(c.ndim))

    @property
    def size(self):
        return self._ptr.contents.size

    @property
    def device(self) -> str:
        """'cpu' or 'gpu'. See also .gpu_backend for which GPU backend."""
        return "gpu" if self._ptr.contents.device == CF_DEVICE_GPU else "cpu"

    @property
    def gpu_backend(self):
        """None, 'cuda', or 'opencl'."""
        b = self._ptr.contents.gpu_backend
        if b == CF_GPU_CUDA:   return "cuda"
        if b == CF_GPU_OPENCL: return "opencl"
        return None

    @property
    def is_view(self) -> bool:
        """True for views like flatten() output — they share their buffer
        with a parent Tensor and cannot be moved between devices."""
        return not bool(self._ptr.contents.owns_data)

    def __repr__(self):
        return (f"neuralc.Tensor(shape={self.shape}, device={self.device!r}"
                f"{', view=True' if self.is_view else ''})")

    # ── ops ──

    def sum(self):
        return float(_lib.tensor_sum(self._ptr))

    def mean(self):
        return float(_lib.tensor_mean(self._ptr))

    def argmax(self):
        return int(_lib.tensor_argmax(self._ptr))

    def fill(self, val):
        _lib.tensor_fill(self._ptr, ctypes.c_float(val))
        return self

    def clone(self):
        return Tensor(_lib.tensor_clone(self._ptr))

    def flatten(self):
        """
        Return a new Tensor that is a *view* onto this tensor's data
        (e.g. collapsing (N, C, H, W) -> (N, C*H*W) ahead of a Dense layer).
        Implemented in C via tensor_reshape(), which sets owns_data=0 on
        the returned struct (see conv.c's flatten() / tensor.c's
        tensor_reshape()) — the buffer itself still belongs to `self`.

        We keep a reference back to `self` on the view (`view._base`) so
        the parent tensor, and therefore the buffer the view points into,
        can't be garbage-collected while the view is still alive.
        """
        ptr = _lib.flatten(self._ptr)
        if not ptr:
            raise MemoryError("flatten failed")
        view = Tensor(ptr, owned=True)
        view._base = self
        return view

    def to(self, device: str):
        """
        Move this tensor's underlying buffer to 'cpu', 'gpu' (auto-picks
        CUDA if available, else OpenCL), 'cuda', or 'opencl'.

        Every branch here exists to dodge a specific cforge_error()->exit(1)
        path in tensor.c:
          - tensor_to_gpu_ex()/tensor_to_cpu() both hard-exit if called on
            a non-owning view (e.g. flatten() output) — checked via
            self.is_view before doing anything else.
          - tensor_to_gpu[_ex]() hard-exits if the requested backend isn't
            compiled in or no device is available at runtime — checked via
            cf_cuda_enabled()/cf_opencl_enabled() first.
        """
        device = device.lower()
        if self.is_view:
            raise RuntimeError(
                "Cannot move a view Tensor (e.g. flatten() output) between "
                "devices — it doesn't own its data buffer. Move the "
                "original/parent tensor instead."
            )

        if device == "cpu":
            if self.device == "cpu":
                return self
            _lib.tensor_to_cpu(self._ptr)
        elif device == "gpu":
            if not (cf_cuda_enabled() or cf_opencl_enabled()):
                raise RuntimeError(
                    "No GPU backend available: libneuralc.so was built "
                    "without -DUSE_CUDA/-DUSE_OPENCL, or no device was "
                    "found at runtime. Check cf_cuda_enabled()/"
                    "cf_opencl_enabled()."
                )
            _lib.tensor_to_gpu(self._ptr)
        elif device == "cuda":
            if not cf_cuda_enabled():
                raise RuntimeError(
                    "CUDA not available (cf_cuda_enabled() is False)."
                )
            _lib.tensor_to_gpu_ex(self._ptr, ctypes.c_int(CF_GPU_CUDA))
        elif device == "opencl":
            if not cf_opencl_enabled():
                raise RuntimeError(
                    "OpenCL not available (cf_opencl_enabled() is False)."
                )
            _lib.tensor_to_gpu_ex(self._ptr, ctypes.c_int(CF_GPU_OPENCL))
        else:
            raise ValueError(
                f"Unknown device: {device!r} (expected 'cpu', 'gpu', "
                f"'cuda', or 'opencl')"
            )
        return self

# ── Dense layer ────────────────────────────────────────────────────────

class Dense:
    def __init__(self, in_features, out_features, activation=ACT_RELU):
        self._ptr = _lib.dense_create(in_features, out_features, activation)
        if not self._ptr:
            raise MemoryError("dense_create failed")

    # Note: ownership transferred to Network on add() — so __del__ is
    # intentionally left out here, matching the original design.

    def param_count(self) -> int:
        return int(_lib.dense_param_count(ctypes.c_void_p(self._ptr)))

    def to(self, device: str):
        """
        Move this layer's weights to 'cpu', 'gpu' (auto-pick), 'cuda', or
        'opencl'. Unlike Conv2D, Dense supports both CUDA and OpenCL
        (layer.c's dense_forward/backward branch on either backend).
        """
        device = device.lower()
        has_gpu_build = cf_cuda_enabled() or cf_opencl_enabled()

        if device == "cpu":
            if not has_gpu_build:
                return self  # never left CPU; dense_to_cpu would hard-exit
            _lib.dense_to_cpu(ctypes.c_void_p(self._ptr))
        elif device == "gpu":
            if not has_gpu_build:
                raise RuntimeError(
                    "No GPU backend available (see cf_cuda_enabled()/"
                    "cf_opencl_enabled())."
                )
            _lib.dense_to_gpu(ctypes.c_void_p(self._ptr))
        elif device == "cuda":
            if not cf_cuda_enabled():
                raise RuntimeError("CUDA not available.")
            _lib.dense_to_gpu_ex(ctypes.c_void_p(self._ptr),
                                 ctypes.c_int(CF_GPU_CUDA))
        elif device == "opencl":
            if not cf_opencl_enabled():
                raise RuntimeError("OpenCL not available.")
            _lib.dense_to_gpu_ex(ctypes.c_void_p(self._ptr),
                                 ctypes.c_int(CF_GPU_OPENCL))
        else:
            raise ValueError(f"Unknown device: {device!r}")
        return self

# ── Conv2D layer ───────────────────────────────────────────────────────

class Conv2D:
    """
    Python wrapper around neuralc's C Conv2D struct (conv.h/conv.c).

    GPU support here is CUDA-only — conv.c has no OpenCL kernels for
    Conv2D/MaxPool2D yet. forward()/backward() actively check that any
    GPU-resident input is specifically CUDA-backed (not OpenCL) before
    calling into C, because feeding an OpenCL-resident tensor into these
    functions would otherwise either hard-exit via cforge_error() (no
    -DUSE_CUDA build) or, per conv.c's own comments on conv2d_to_gpu,
    risk silently corrupting memory rather than erroring cleanly.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self._ptr = _lib.conv2d_create(
            ctypes.c_int(in_channels), ctypes.c_int(out_channels),
            ctypes.c_int(kernel_size), ctypes.c_int(stride), ctypes.c_int(pad))
        if not self._ptr:
            raise MemoryError("conv2d_create failed")

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.pad          = pad

    # Note: like Dense, no __del__ — ownership transfers to a Network
    # sequence layer via Network.add().

    @staticmethod
    def _require_cuda_if_gpu(t: Tensor, label: str):
        if t.device == "gpu" and t.gpu_backend != "cuda":
            raise RuntimeError(
                f"Conv2D/MaxPool2D only support CUDA-resident GPU tensors "
                f"(no OpenCL kernels yet), but {label} is on "
                f"{t.gpu_backend!r}. Move it explicitly with "
                f".to('cuda') rather than .to('gpu'), which may have "
                f"picked OpenCL."
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Expects x of shape (batch, in_channels, H, W). Allocates and
        returns a correctly-shaped, correctly-device-matched output
        tensor automatically.
        """
        self._require_cuda_if_gpu(x, "input")
        batch, channels, h, w = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Conv2D expected {self.in_channels} input channels, got {channels}"
            )
        out_h = _out_dim(h, self.kernel_size, self.stride, self.pad)
        out_w = _out_dim(w, self.kernel_size, self.stride, self.pad)
        out = Tensor.zeros([batch, self.out_channels, out_h, out_w])
        # conv2d_forward does not allocate `output` itself — it must
        # already be resident on the same device as `x` (Tensor.zeros()
        # always creates CPU tensors, so move it explicitly if needed).
        if x.device == "gpu":
            out.to("cuda")
        _lib.conv2d_forward(ctypes.c_void_p(self._ptr), x._ptr, out._ptr)
        return out

    def backward(self, grad_out: Tensor, input_shape) -> Tensor:
        """
        Backpropagate gradients. `input_shape` is the shape of the tensor
        originally passed to forward(). Must be called after a matching
        forward() (conv2d_backward reads a cache populated there).
        Returns grad_in, the gradient w.r.t. that input.
        """
        self._require_cuda_if_gpu(grad_out, "grad_out")
        grad_in = Tensor.zeros(list(input_shape))
        if grad_out.device == "gpu":
            grad_in.to("cuda")
        _lib.conv2d_backward(ctypes.c_void_p(self._ptr), grad_out._ptr, grad_in._ptr)
        return grad_in

    def param_count(self) -> int:
        return int(_lib.conv2d_param_count(ctypes.c_void_p(self._ptr)))

    def to(self, device: str):
        """Move this layer's weights to 'cpu' or 'cuda' ('gpu' is
        accepted as an alias for 'cuda' — Conv2D has no OpenCL path)."""
        device = device.lower()
        if device == "cpu":
            if not cf_cuda_enabled():
                return self  # never left CPU; conv2d_to_cpu would hard-exit
            _lib.conv2d_to_cpu(ctypes.c_void_p(self._ptr))
        elif device in ("gpu", "cuda"):
            if not cf_cuda_enabled():
                raise RuntimeError(
                    "Conv2D/MaxPool2D GPU support is CUDA-only (no OpenCL "
                    "kernels yet), and cf_cuda_enabled() is False — either "
                    "this build lacks -DUSE_CUDA or no CUDA device was "
                    "found at runtime."
                )
            _lib.conv2d_to_gpu(ctypes.c_void_p(self._ptr))
        elif device == "opencl":
            raise RuntimeError(
                "Conv2D/MaxPool2D has no OpenCL kernels yet (CUDA only)."
            )
        else:
            raise ValueError(f"Unknown device: {device!r} (expected 'cpu' or 'cuda')")
        return self

# ── MaxPool2D layer ──────────────────────────────────────────────────

class MaxPool2D:
    """
    Python wrapper around neuralc's maxpool2d_forward/backward (conv.c).

    No learnable parameters, so no underlying C layer object — forward/
    backward operate directly on Tensors. Per conv.c: `mask` is shaped
    like the *output* (NOT the input), and for each output position it
    stores the flat index of the winning element within the *input*
    tensor's data buffer — backward() is then a pure scatter over that
    index. The mask is generated and cached internally on forward(), so
    callers never need to manage it themselves. CUDA-only, same as
    Conv2D — see Conv2D's class docstring.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride    = stride
        self._mask        = None
        self._input_shape = None

    def forward(self, x: Tensor) -> Tensor:
        Conv2D._require_cuda_if_gpu(x, "input")
        batch, channels, h, w = x.shape
        out_h = _out_dim(h, self.pool_size, self.stride, 0)
        out_w = _out_dim(w, self.pool_size, self.stride, 0)

        out  = Tensor.zeros([batch, channels, out_h, out_w])
        # mask shape matches the OUTPUT (per conv.h: "mask [same shape as
        # output]"), not the input — it holds one input-flat-index per
        # output position, not a per-window offset.
        mask = Tensor.zeros([batch, channels, out_h, out_w])
        if x.device == "gpu":
            out.to("cuda")
            mask.to("cuda")

        _lib.maxpool2d_forward(x._ptr, out._ptr, mask._ptr,
                               ctypes.c_int(self.pool_size), ctypes.c_int(self.stride))

        self._mask        = mask
        self._input_shape = x.shape
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        if self._mask is None:
            raise RuntimeError(
                "MaxPool2D.backward() called before a matching forward(); "
                "no pooling mask is available."
            )
        Conv2D._require_cuda_if_gpu(grad_out, "grad_out")
        grad_in = Tensor.zeros(list(self._input_shape))
        if grad_out.device == "gpu":
            grad_in.to("cuda")
        _lib.maxpool2d_backward(grad_out._ptr, grad_in._ptr, self._mask._ptr,
                                ctypes.c_int(self.pool_size), ctypes.c_int(self.stride))
        return grad_in

# ── Network ────────────────────────────────────────────────────────────

class Network:
    def __init__(self):
        self._ptr     = _lib.nn_create()
        self._pred    = None
        self._grad    = None
        self._out_dim = None
        if not self._ptr:
            raise MemoryError("nn_create failed")

    def __del__(self):
        if self._ptr:
            _lib.nn_free(ctypes.c_void_p(self._ptr))
            self._ptr = None
        # Don't call tensor_free() here directly: self._pred/self._grad
        # are Tensor objects that already own+free their buffers in their
        # own __del__. Just drop the references and let that run exactly
        # once — manually freeing here too would double-free.
        self._pred = None
        self._grad = None

    def add(self, layer):
        _lib.nn_add_layer(ctypes.c_void_p(self._ptr),
                          ctypes.c_void_p(layer._ptr))
        self._out_dim = None

    def _ensure_buffers(self, batch, out_dim):
        if (self._pred is None or
            self._pred.shape != (batch, out_dim)):
            self._pred = Tensor.zeros([batch, out_dim])
            self._grad = Tensor.zeros([batch, out_dim])

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        if self._out_dim is None:
            raise RuntimeError(
                "Call net.set_output_dim(n) before first forward, "
                "or use train_step() which sets it automatically."
            )
        self._ensure_buffers(batch, self._out_dim)
        _lib.nn_forward(ctypes.c_void_p(self._ptr),
                        x._ptr, self._pred._ptr)
        return self._pred.clone()

    def set_output_dim(self, n: int):
        """Set the output dimension (number of neurons in last layer)."""
        self._out_dim = n

    def train_step(self, x: Tensor, y: Tensor,
                   loss_type: int = LOSS_BINARY_CROSS) -> float:
        batch   = x.shape[0]
        out_dim = y.shape[1] if len(y.shape) > 1 else 1
        self._out_dim = out_dim
        self._ensure_buffers(batch, out_dim)
        loss = _lib.nn_train_step(
            ctypes.c_void_p(self._ptr),
            x._ptr, y._ptr,
            ctypes.c_int(loss_type),
            self._pred._ptr,
            self._grad._ptr
        )
        return float(loss)

    def save(self, path: str) -> int:
        return int(_lib.nn_save(ctypes.c_void_p(self._ptr), path.encode()))

    def load(self, path: str) -> int:
        return int(_lib.nn_load(ctypes.c_void_p(self._ptr), path.encode()))

    def accuracy_binary(self, pred: Tensor, target: Tensor) -> float:
        return float(_lib.nn_accuracy_binary(pred._ptr, target._ptr))

    def accuracy_multiclass(self, pred: Tensor, target: Tensor) -> float:
        return float(_lib.nn_accuracy_multiclass(pred._ptr, target._ptr))

# ── Optimizers ─────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self._ptr = _lib.adam_create(
            ctypes.c_float(lr),   ctypes.c_float(beta1),
            ctypes.c_float(beta2),ctypes.c_float(eps),
            ctypes.c_float(weight_decay))
        if not self._ptr: raise MemoryError("adam_create failed")

    def __del__(self):
        if self._ptr:
            _lib.adam_free(ctypes.c_void_p(self._ptr))
            self._ptr = None

    def step(self, net: Network):
        _lib.adam_step(ctypes.c_void_p(self._ptr), ctypes.c_void_p(net._ptr))

class SGD:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self._ptr = _lib.sgd_create(
            ctypes.c_float(lr), ctypes.c_float(momentum),
            ctypes.c_float(weight_decay))
        if not self._ptr: raise MemoryError("sgd_create failed")

    def __del__(self):
        if self._ptr:
            _lib.sgd_free(ctypes.c_void_p(self._ptr))
            self._ptr = None

    def step(self, net: Network):
        _lib.sgd_step(ctypes.c_void_p(self._ptr), ctypes.c_void_p(net._ptr))

class RMSProp:
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8, weight_decay=0.0):
        self._ptr = _lib.rmsprop_create(
            ctypes.c_float(lr),  ctypes.c_float(rho),
            ctypes.c_float(eps), ctypes.c_float(weight_decay))
        if not self._ptr: raise MemoryError("rmsprop_create failed")

    def __del__(self):
        if self._ptr:
            _lib.rmsprop_free(ctypes.c_void_p(self._ptr))
            self._ptr = None

    def step(self, net: Network):
        _lib.rmsprop_step(ctypes.c_void_p(self._ptr), ctypes.c_void_p(net._ptr))

# ── Quick usage example (run as script) ────────────────────────────────

if __name__ == "__main__":
    print("neuralc Python bindings loaded!")
    print(f"Library: {_find_lib()}")
    print(f"CUDA available:   {cf_cuda_enabled()}")
    print(f"OpenCL available: {cf_opencl_enabled()}")

    X_np = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y_np = np.array([[0],[1],[1],[0]],          dtype=np.float32)

    X = Tensor.from_numpy(X_np)
    Y = Tensor.from_numpy(Y_np)

    net = Network()
    net.add(Dense(2, 8,  ACT_RELU))
    net.add(Dense(8, 1,  ACT_SIGMOID))
    net.set_output_dim(1)

    opt = Adam(lr=0.01)

    print("\nTraining XOR...")
    for epoch in range(3001):
        loss = net.train_step(X, Y, LOSS_BINARY_CROSS)
        opt.step(net)
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}  loss={loss:.6f}")

    net.save("xor_from_python.bin")
    print("\nWeights saved to xor_from_python.bin")
    print("Done! neuralc Python bindings working.")

    if hasattr(_lib, "conv2d_create"):
        print("\nSmoke-testing Conv2D + MaxPool2D + flatten (CPU)...")
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
        pool = MaxPool2D(pool_size=2, stride=2)

        img = Tensor.zeros([2, 1, 8, 8])
        fmap = conv.forward(img)
        print(f"  conv output shape:   {fmap.shape}")
        pooled = pool.forward(fmap)
        print(f"  pooled output shape: {pooled.shape}")
        flat = pooled.flatten()
        print(f"  flattened shape:     {flat.shape}")
        print(f"  conv2d param_count:  {conv.param_count()}")

        if cf_cuda_enabled():
            print("\nSmoke-testing Conv2D on CUDA...")
            conv.to("cuda")
            img_gpu = Tensor.zeros([2, 1, 8, 8]).to("cuda")
            fmap_gpu = conv.forward(img_gpu)
            print(f"  conv (cuda) output shape: {fmap_gpu.shape}, "
                  f"device={fmap_gpu.device}, backend={fmap_gpu.gpu_backend}")
