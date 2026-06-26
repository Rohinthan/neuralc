"""
neuralc.py — Python bindings for the neuralc C library via ctypes.

Usage:
    from neuralc import Tensor, Network, Dense, Adam, ACT_RELU, ACT_SIGMOID

Requirements:
    Build the shared library first:
        make libneuralc

Example:
    import neuralc as nc

    # Create tensors
    x = nc.Tensor.zeros([4, 2])
    y = nc.Tensor.zeros([4, 1])

    # Build network
    net = nc.Network()
    net.add(nc.Dense(2, 8,  nc.ACT_RELU))
    net.add(nc.Dense(8, 1,  nc.ACT_SIGMOID))

    # Train
    opt = nc.Adam(lr=0.01)
    for epoch in range(3000):
        loss = net.train_step(x, y, nc.LOSS_BINARY_CROSS)
        opt.step(net)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}  loss={loss:.6f}")

    # Predict
    pred = net.forward(x)
    print("Accuracy:", net.accuracy_binary(pred, y))
"""

import ctypes
import os
import sys
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
        "libneuralc.so not found. Run 'make libneuralc' first.\n"
        "Searched: " + str(candidates)
    )

_lib = ctypes.CDLL(_find_lib())

# ── C struct mirrors ───────────────────────────────────────────────────

class _CTensor(ctypes.Structure):
    _fields_ = [
        ("data",      ctypes.POINTER(ctypes.c_float)),
        ("shape",     ctypes.c_int * 8),
        ("ndim",      ctypes.c_int),
        ("size",      ctypes.c_size_t),
        ("owns_data", ctypes.c_int),
    ]

# ── activation / loss enums ────────────────────────────────────────────

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

    # tensor lifecycle
    _lib.tensor_create.restype  = P(_CTensor)
    _lib.tensor_create.argtypes = [P(ctypes.c_int), ctypes.c_int]
    _lib.tensor_zeros.restype   = P(_CTensor)
    _lib.tensor_zeros.argtypes  = [P(ctypes.c_int), ctypes.c_int]
    _lib.tensor_free.restype    = None
    _lib.tensor_free.argtypes   = [P(_CTensor)]
    _lib.tensor_clone.restype   = P(_CTensor)
    _lib.tensor_clone.argtypes  = [P(_CTensor)]

    # tensor ops
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

    # nn
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
    _lib.nn_accuracy_binary.restype     = ctypes.c_float
    _lib.nn_accuracy_binary.argtypes    = [P(_CTensor), P(_CTensor)]
    _lib.nn_accuracy_multiclass.restype  = ctypes.c_float
    _lib.nn_accuracy_multiclass.argtypes = [P(_CTensor), P(_CTensor)]

    # dense layer
    _lib.dense_create.restype   = ctypes.c_void_p
    _lib.dense_create.argtypes  = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

    # adam
    _lib.adam_create.restype    = ctypes.c_void_p
    _lib.adam_create.argtypes   = [ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _lib.adam_step.restype      = None
    _lib.adam_step.argtypes     = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.adam_free.restype      = None
    _lib.adam_free.argtypes     = [ctypes.c_void_p]

    # sgd
    _lib.sgd_create.restype     = ctypes.c_void_p
    _lib.sgd_create.argtypes    = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _lib.sgd_step.restype       = None
    _lib.sgd_step.argtypes      = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.sgd_free.restype       = None
    _lib.sgd_free.argtypes      = [ctypes.c_void_p]

    # rmsprop
    _lib.rmsprop_create.restype     = ctypes.c_void_p
    _lib.rmsprop_create.argtypes    = [ctypes.c_float, ctypes.c_float,
                                       ctypes.c_float, ctypes.c_float]
    _lib.rmsprop_step.restype       = None
    _lib.rmsprop_step.argtypes      = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.rmsprop_free.restype       = None
    _lib.rmsprop_free.argtypes      = [ctypes.c_void_p]

_setup_signatures()

# ── helper: shape list → c_int array ──────────────────────────────────

def _shape_arr(shape):
    arr = (ctypes.c_int * len(shape))(*shape)
    return arr

# ── Tensor class ───────────────────────────────────────────────────────

class Tensor:
    """Python wrapper around neuralc's C Tensor struct."""

    def __init__(self, ptr, owned=True):
        self._ptr   = ptr
        self._owned = owned

    def __del__(self):
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
        # copy data
        src = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes.memmove(t._ptr.contents.data, src,
                       array.size * ctypes.sizeof(ctypes.c_float))
        return t

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy float32 array."""
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

    def __repr__(self):
        return f"neuralc.Tensor(shape={self.shape})"

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

# ── Dense layer ────────────────────────────────────────────────────────

class Dense:
    def __init__(self, in_features, out_features, activation=ACT_RELU):
        self._ptr = _lib.dense_create(in_features, out_features, activation)
        if not self._ptr:
            raise MemoryError("dense_create failed")

    # Note: ownership transferred to Network on add()
    # so __del__ is intentionally left out here

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
        if self._pred: _lib.tensor_free(self._pred._ptr); self._pred = None
        if self._grad: _lib.tensor_free(self._grad._ptr); self._grad = None

    def add(self, layer: Dense):
        _lib.nn_add_layer(ctypes.c_void_p(self._ptr),
                          ctypes.c_void_p(layer._ptr))
        self._out_dim = None   # reset cache

    def _ensure_buffers(self, batch, out_dim):
        if (self._pred is None or
            self._pred.shape != (batch, out_dim)):
            self._pred = Tensor.zeros([batch, out_dim])
            self._grad = Tensor.zeros([batch, out_dim])

    def forward(self, x: Tensor) -> Tensor:
        # determine output size from last layer output
        # we call forward and return the result
        batch = x.shape[0]
        # Use a temporary — determine out_dim lazily on first call
        if self._out_dim is None:
            # Can't know output size without running; use 1 to probe then resize
            # Instead, user should pass out_dim on first call OR we
            # require add() to record it. For simplicity, infer after forward.
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
        return int(_lib.nn_save(ctypes.c_void_p(self._ptr),
                                path.encode()))

    def load(self, path: str) -> int:
        return int(_lib.nn_load(ctypes.c_void_p(self._ptr),
                                path.encode()))

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
        _lib.adam_step(ctypes.c_void_p(self._ptr),
                       ctypes.c_void_p(net._ptr))

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
        _lib.sgd_step(ctypes.c_void_p(self._ptr),
                      ctypes.c_void_p(net._ptr))

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
        _lib.rmsprop_step(ctypes.c_void_p(self._ptr),
                          ctypes.c_void_p(net._ptr))

# ── Quick usage example (run as script) ────────────────────────────────

if __name__ == "__main__":
    print("neuralc Python bindings loaded!")
    print(f"Library: {_find_lib()}")

    import numpy as np

    # XOR dataset
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
