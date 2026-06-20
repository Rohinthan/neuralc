#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn.h"

/* ── SGD (+ optional momentum) ──────────────────────────────────── */
typedef struct {
    float      lr;          /* learning rate           */
    float      momentum;    /* 0.0 = vanilla SGD       */
    float      weight_decay;
    /* velocity buffers (one per layer W and b) */
    Tensor    *vW[NN_MAX_LAYERS];
    Tensor    *vb[NN_MAX_LAYERS];
    int        initialized;
} SGD;

SGD  *sgd_create(float lr, float momentum, float weight_decay);
void  sgd_step(SGD *opt, Network *net);
void  sgd_free(SGD *opt);

/* ── Adam ───────────────────────────────────────────────────────── */
typedef struct {
    float      lr;
    float      beta1;       /* 0.9   */
    float      beta2;       /* 0.999 */
    float      eps;         /* 1e-8  */
    float      weight_decay;
    int        t;           /* step counter */
    /* first and second moment buffers */
    Tensor    *mW[NN_MAX_LAYERS];
    Tensor    *vW[NN_MAX_LAYERS];
    Tensor    *mb[NN_MAX_LAYERS];
    Tensor    *vb[NN_MAX_LAYERS];
    int        initialized;
} Adam;

Adam *adam_create(float lr, float beta1, float beta2, float eps,
                  float weight_decay);
void  adam_step(Adam *opt, Network *net);
void  adam_free(Adam *opt);

/* ── learning rate scheduler ────────────────────────────────────── */
void  lr_step_decay(SGD *opt, int epoch, int decay_every, float decay_rate);
void  lr_cosine(SGD *opt, int epoch, int total_epochs, float lr_min,
                float lr_max);

/* ── RMSProp ────────────────────────────────────────────────────── */
typedef struct {
    float   lr;
    float   rho;         /* decay factor, typically 0.9             */
    float   eps;         /* numerical stability e.g. 1e-8           */
    float   weight_decay;
    Tensor *vW[NN_MAX_LAYERS];   /* running mean square of grad W   */
    Tensor *vb[NN_MAX_LAYERS];   /* running mean square of grad b   */
    int     initialized;
} RMSProp;

RMSProp *rmsprop_create(float lr, float rho, float eps, float weight_decay);
void     rmsprop_step(RMSProp *opt, Network *net);
void     rmsprop_free(RMSProp *opt);

#endif /* OPTIMIZER_H */
