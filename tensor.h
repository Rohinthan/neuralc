#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* Maximum number of dimensions supported */
#define TENSOR_MAX_DIMS 8

typedef struct {
    float  *data;           /* flat row-major data buffer          */
    int     shape[TENSOR_MAX_DIMS];
    int     ndim;           /* number of dimensions                */
    size_t  size;           /* total number of elements            */
    int     owns_data;      /* 1 = free data on tensor_free()      */
} Tensor;

/* ── lifecycle ─────────────────────────────────────────────────── */
Tensor *tensor_create(const int *shape, int ndim);
Tensor *tensor_zeros(const int *shape, int ndim);
Tensor *tensor_ones(const int *shape, int ndim);
Tensor *tensor_rand(const int *shape, int ndim);   /* uniform [0,1) */
Tensor *tensor_randn(const int *shape, int ndim);  /* N(0,1)        */
Tensor *tensor_clone(const Tensor *t);
void    tensor_free(Tensor *t);

/* ── element access ────────────────────────────────────────────── */
/* E.g. tensor_at2(t, i, j)  →  t->data[i*cols + j]               */
float  tensor_get(const Tensor *t, const int *idx);
void   tensor_set(Tensor *t, const int *idx, float val);

/* Convenience macros for 1-D / 2-D access */
#define T1(t, i)      ((t)->data[(i)])
#define T2(t, i, j)   ((t)->data[(i)*(t)->shape[1] + (j)])

/* ── element-wise ops (result written to out; out may == a or b) ─ */
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);  /* Hadamard */
void tensor_div(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_scale(const Tensor *a, float s, Tensor *out);
void tensor_add_scalar(const Tensor *a, float s, Tensor *out);
void tensor_neg(const Tensor *a, Tensor *out);
void tensor_abs(const Tensor *a, Tensor *out);
void tensor_square(const Tensor *a, Tensor *out);
void tensor_sqrt_t(const Tensor *a, Tensor *out);
void tensor_exp(const Tensor *a, Tensor *out);
void tensor_log_t(const Tensor *a, Tensor *out);
void tensor_clip(const Tensor *a, float lo, float hi, Tensor *out);

/* ── reductions ────────────────────────────────────────────────── */
float tensor_sum(const Tensor *t);
float tensor_mean(const Tensor *t);
float tensor_max(const Tensor *t);
float tensor_min(const Tensor *t);
float tensor_norm(const Tensor *t);   /* L2 norm */

/* ── matrix ops (2-D tensors) ──────────────────────────────────── */
void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_transpose(const Tensor *a, Tensor *out);  /* 2-D only   */

/* ── activation functions (element-wise, in-place safe) ────────── */
void tensor_relu(const Tensor *a, Tensor *out);
void tensor_relu_grad(const Tensor *a, const Tensor *grad, Tensor *out);
void tensor_sigmoid(const Tensor *a, Tensor *out);
void tensor_sigmoid_grad(const Tensor *sig, const Tensor *grad, Tensor *out);
void tensor_tanh_t(const Tensor *a, Tensor *out);
void tensor_tanh_grad(const Tensor *th, const Tensor *grad, Tensor *out);
void tensor_softmax(const Tensor *a, Tensor *out);  /* row-wise for 2-D */

/* ── shape ops ──────────────────────────────────────────────────── */
Tensor *tensor_reshape(Tensor *t, const int *new_shape, int new_ndim);
void    tensor_fill(Tensor *t, float val);
void    tensor_copy_data(Tensor *dst, const Tensor *src);

/* ── utilities ──────────────────────────────────────────────────── */
void tensor_print(const Tensor *t, const char *name);
int  tensor_shape_equal(const Tensor *a, const Tensor *b);

#endif /* TENSOR_H */
