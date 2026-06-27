#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
/* assert.h replaced by CF_CHECK */
#include <time.h>

#ifdef USE_OMP
#include <omp.h>
#endif

/* ── helpers ─────────────────────────────────────────────────────── */

static size_t shape_size(const int *shape, int ndim) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
    return n;
}

/* Box-Muller: standard normal sample */
static float randn_sample(void) {
    static int   has_spare = 0;
    static float spare;
    if (has_spare) { has_spare = 0; return spare; }
    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u*u + v*v;
    } while (s >= 1.0f || s == 0.0f);
    float mul = sqrtf(-2.0f * logf(s) / s);
    spare = v * mul;
    has_spare = 1;
    return u * mul;
}

/* ── lifecycle ───────────────────────────────────────────────────── */

Tensor *tensor_create(const int *shape, int ndim) {
    CF_CHECK(ndim > 0 && ndim <= TENSOR_MAX_DIMS, "ndim out of range [1..TENSOR_MAX_DIMS]");
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->ndim = ndim;
    t->size = shape_size(shape, ndim);
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->data = (float *)malloc(t->size * sizeof(float));
    t->owns_data = 1;
    if (!t->data) { free(t); return NULL; }
    return t;
}

Tensor *tensor_zeros(const int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (t) memset(t->data, 0, t->size * sizeof(float));
    return t;
}

Tensor *tensor_ones(const int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
}

Tensor *tensor_rand(const int *shape, int ndim) {
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = (float)rand() / RAND_MAX;
    return t;
}

Tensor *tensor_randn(const int *shape, int ndim) {
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = randn_sample();
    return t;
}

Tensor *tensor_clone(const Tensor *t) {
    Tensor *c = tensor_create(t->shape, t->ndim);
    if (c) memcpy(c->data, t->data, t->size * sizeof(float));
    return c;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->owns_data) free(t->data);
    free(t);
}

/* ── element access ──────────────────────────────────────────────── */

float tensor_get(const Tensor *t, const int *idx) {
    size_t off = 0, stride = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        off    += idx[d] * stride;
        stride *= t->shape[d];
    }
    return t->data[off];
}

void tensor_set(Tensor *t, const int *idx, float val) {
    size_t off = 0, stride = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        off    += idx[d] * stride;
        stride *= t->shape[d];
    }
    t->data[off] = val;
}

/* ── element-wise ops ────────────────────────────────────────────── */

#ifdef USE_OMP
  #define OMP_ELWISE _Pragma("omp parallel for schedule(static)")
#else
  #define OMP_ELWISE
#endif

#define ELWISE2(name, expr)                                              \
void name(const Tensor *a, const Tensor *b, Tensor *out) {              \
    CF_CHECK_SHAPE(a, out); CF_CHECK_SHAPE(b, out);                      \
    OMP_ELWISE                                                           \
    for (size_t i = 0; i < a->size; i++)                                 \
        out->data[i] = (expr);                                           \
}

ELWISE2(tensor_add, a->data[i] + b->data[i])
ELWISE2(tensor_sub, a->data[i] - b->data[i])
ELWISE2(tensor_mul, a->data[i] * b->data[i])
ELWISE2(tensor_div, a->data[i] / b->data[i])

void tensor_scale(const Tensor *a, float s, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] * s;
}
void tensor_add_scalar(const Tensor *a, float s, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] + s;
}
void tensor_neg(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = -a->data[i];
}
void tensor_abs(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = fabsf(a->data[i]);
}
void tensor_square(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] * a->data[i];
}
void tensor_sqrt_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = sqrtf(a->data[i]);
}
void tensor_exp(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = expf(a->data[i]);
}
void tensor_log_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = logf(a->data[i]);
}
void tensor_clip(const Tensor *a, float lo, float hi, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) {
        float v = a->data[i];
        out->data[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

/* ── reductions ──────────────────────────────────────────────────── */

float tensor_sum(const Tensor *t) {
    float s = 0.0f;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:s) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 0; i < t->size; i++) s += t->data[i];
    return s;
}
float tensor_mean(const Tensor *t) { return tensor_sum(t) / (float)t->size; }
float tensor_max(const Tensor *t) {
    float m = t->data[0];
#ifdef USE_OMP
    #pragma omp parallel for reduction(max:m) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 1; i < t->size; i++) if (t->data[i] > m) m = t->data[i];
    return m;
}
float tensor_min(const Tensor *t) {
    float m = t->data[0];
#ifdef USE_OMP
    #pragma omp parallel for reduction(min:m) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 1; i < t->size; i++) if (t->data[i] < m) m = t->data[i];
    return m;
}
float tensor_norm(const Tensor *t) {
    float s = 0.0f;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:s) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 0; i < t->size; i++) s += t->data[i] * t->data[i];
    return sqrtf(s);
}

/* ── matrix ops ──────────────────────────────────────────────────── */

void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out) {
    /* a: [M,K]  b: [K,N]  out: [M,N] */
    CF_CHECK(a->ndim==2 && b->ndim==2 && out->ndim==2,
             "matmul: all tensors must be 2D");
    int M = a->shape[0], K = a->shape[1];
    int K2 = b->shape[0], N = b->shape[1];
    CF_CHECK(K == K2, "matmul: inner dimensions must match (a.cols == b.rows)");
    CF_CHECK(out->shape[0]==M && out->shape[1]==N,
             "matmul: output shape mismatch");

    /*
     * Restructured to i,j outer / k inner so collapse(2) is safe.
     * Each (i,j) pair has its own local `sum` — no data race.
     * Original i,k,j order was NOT parallel-safe for j dimension.
     */
#ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static) \
        if((size_t)M*K*N > 8192)
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += a->data[i*K + k] * b->data[k*N + j];
            out->data[i*N + j] = sum;   /* single write — no race */
        }
    }
}

void tensor_transpose(const Tensor *a, Tensor *out) {
    CF_CHECK(a->ndim==2 && out->ndim==2, "transpose: tensors must be 2D");
    int R = a->shape[0], C = a->shape[1];
    CF_CHECK(out->shape[0]==C && out->shape[1]==R, "transpose: output shape must be [cols, rows]");
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            out->data[j*R + i] = a->data[i*C + j];
}

/* ── activations ─────────────────────────────────────────────────── */

void tensor_relu(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
}
void tensor_relu_grad(const Tensor *a, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(a, grad); CF_CHECK_SHAPE(a, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] > 0.0f ? grad->data[i] : 0.0f;
}
void tensor_sigmoid(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
}
void tensor_sigmoid_grad(const Tensor *sig, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(sig, grad); CF_CHECK_SHAPE(sig, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(sig->size > 4096)
#endif
    for (size_t i = 0; i < sig->size; i++) {
        float s = sig->data[i];
        out->data[i] = grad->data[i] * s * (1.0f - s);
    }
}
void tensor_tanh_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = tanhf(a->data[i]);
}
void tensor_tanh_grad(const Tensor *th, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(th, grad); CF_CHECK_SHAPE(th, out);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(th->size > 4096)
#endif
    for (size_t i = 0; i < th->size; i++) {
        float t = th->data[i];
        out->data[i] = grad->data[i] * (1.0f - t*t);
    }
}

/* row-wise softmax for 2-D [batch, classes] */
void tensor_softmax(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    int rows = (a->ndim >= 2) ? a->shape[0] : 1;
    int cols = (int)(a->size / rows);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(rows > 16)
#endif
    for (int r = 0; r < rows; r++) {
        float *src = a->data   + r*cols;
        float *dst = out->data + r*cols;
        float mx = src[0];
        for (int c = 1; c < cols; c++) if (src[c] > mx) mx = src[c];
        float s = 0.0f;
        for (int c = 0; c < cols; c++) { dst[c] = expf(src[c]-mx); s += dst[c]; }
        for (int c = 0; c < cols; c++) dst[c] /= s;
    }
}

/* ── shape ops ───────────────────────────────────────────────────── */

Tensor *tensor_reshape(Tensor *t, const int *new_shape, int new_ndim) {
    size_t new_size = shape_size(new_shape, new_ndim);
    CF_CHECK(new_size == t->size, "reshape: total elements must stay the same");
    Tensor *r = (Tensor *)malloc(sizeof(Tensor));
    if (!r) return NULL;
    r->data      = t->data;
    r->size      = t->size;
    r->ndim      = new_ndim;
    r->owns_data = 0;   /* view: does not own the data */
    memcpy(r->shape, new_shape, new_ndim * sizeof(int));
    return r;
}

void tensor_fill(Tensor *t, float val) {
    for (size_t i = 0; i < t->size; i++) t->data[i] = val;
}

void tensor_copy_data(Tensor *dst, const Tensor *src) {
    CF_CHECK(dst->size == src->size, "copy_data: size mismatch");
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

/* ── utilities ───────────────────────────────────────────────────── */

void tensor_print(const Tensor *t, const char *name) {
    printf("Tensor '%s'  shape=[", name ? name : "?");
    for (int d = 0; d < t->ndim; d++) {
        printf("%d", t->shape[d]);
        if (d < t->ndim-1) printf(",");
    }
    printf("]  size=%zu\n", t->size);
    if (t->ndim == 1) {
        for (size_t i = 0; i < t->size; i++) printf("  %.4f", t->data[i]);
        printf("\n");
    } else if (t->ndim == 2) {
        int R = t->shape[0], C = t->shape[1];
        for (int i = 0; i < R; i++) {
            printf("  [");
            for (int j = 0; j < C; j++) printf(" %8.4f", t->data[i*C+j]);
            printf(" ]\n");
        }
    } else {
        /* flat dump for higher dims */
        for (size_t i = 0; i < t->size; i++) printf("  %.4f", t->data[i]);
        printf("\n");
    }
}

int tensor_shape_equal(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int d = 0; d < a->ndim; d++)
        if (a->shape[d] != b->shape[d]) return 0;
    return 1;
}

/* ── argmax / argmin ─────────────────────────────────────────────── */

int tensor_argmax(const Tensor *t) {
    int best = 0;
    float best_val = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] > best_val) { best_val = t->data[i]; best = (int)i; }
    return best;
}

int tensor_argmin(const Tensor *t) {
    int best = 0;
    float best_val = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] < best_val) { best_val = t->data[i]; best = (int)i; }
    return best;
}

/* per-row argmax for [batch, classes] — fills caller-supplied int array */
void tensor_argmax_rows(const Tensor *t, int *out) {
    CF_CHECK_2D(t);
    int rows = t->shape[0], cols = t->shape[1];
    for (int r = 0; r < rows; r++) {
        int   best     = 0;
        float best_val = t->data[r * cols];
        for (int c = 1; c < cols; c++) {
            float v = t->data[r * cols + c];
            if (v > best_val) { best_val = v; best = c; }
        }
        out[r] = best;
    }
}

/* ── axis reductions ─────────────────────────────────────────────── */

void tensor_sum_axis(const Tensor *t, int axis, Tensor *out) {
    CF_CHECK_2D(t);
    int rows = t->shape[0], cols = t->shape[1];
    tensor_fill(out, 0.0f);
    if (axis == 0) {
        /* sum across rows → out shape [1, cols] */
        CF_CHECK(out->size == (size_t)cols,
                 "sum_axis(0): out must have size == cols");
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                out->data[c] += t->data[r * cols + c];
    } else {
        /* sum across cols → out shape [rows, 1] */
        CF_CHECK(out->size == (size_t)rows,
                 "sum_axis(1): out must have size == rows");
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                out->data[r] += t->data[r * cols + c];
    }
}

void tensor_mean_axis(const Tensor *t, int axis, Tensor *out) {
    tensor_sum_axis(t, axis, out);
    float div = (axis == 0) ? (float)t->shape[0] : (float)t->shape[1];
    tensor_scale(out, 1.0f / div, out);
}

/* ── error handler ───────────────────────────────────────────────── */

void cforge_error(const char *msg, const char *file, int line) {
    fprintf(stderr, "\n[cforge ERROR] %s\n  at %s:%d\n", msg, file, line);
    exit(1);
}
