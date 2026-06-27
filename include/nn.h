#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "layer.h"

#define NN_MAX_LAYERS 32

/* ── loss functions ─────────────────────────────────────────────── */
typedef enum {
    LOSS_MSE             = 0,   /* mean squared error          */
    LOSS_BINARY_CROSS    = 1,   /* binary cross-entropy        */
    LOSS_CROSS_ENTROPY   = 2    /* softmax + cross-entropy     */
} LossType;

/* ── network ────────────────────────────────────────────────────── */
typedef struct {
    DenseLayer *layers[NN_MAX_LAYERS];
    int         num_layers;

    /* cached intermediate outputs for each layer */
    Tensor     *outputs[NN_MAX_LAYERS];   /* [batch, out_f] per layer */
} Network;

/* ── lifecycle ──────────────────────────────────────────────────── */
Network *nn_create(void);
void     nn_add_layer(Network *net, DenseLayer *layer);
void     nn_free(Network *net);

/* ── forward / backward / loss ──────────────────────────────────── */
/*
 * nn_forward:
 *   input   [batch, in_features of first layer]
 *   output  [batch, out_features of last layer]  (caller allocates)
 */
void  nn_forward(Network *net, const Tensor *input, Tensor *output);

/*
 * nn_loss:  compute scalar loss + initial gradient
 *   pred    [batch, out] — network output
 *   target  [batch, out] — ground truth
 *   grad    [batch, out] — dL/d(pred) written here (caller allocates)
 *   returns scalar loss value
 */
float nn_loss(LossType type, const Tensor *pred, const Tensor *target,
              Tensor *grad);

/*
 * nn_backward:
 *   grad_out  [batch, out] — gradient of loss w.r.t. last layer output
 *   Runs backprop through all layers; populates dW, db, dX per layer.
 */
void  nn_backward(Network *net, const Tensor *grad_out);

/* ── convenience: single training step ─────────────────────────── */
/*
 * Returns the scalar loss value.
 * Does NOT update weights (call optimizer after this).
 */
float nn_train_step(Network *net, const Tensor *input, const Tensor *target,
                    LossType loss_type, Tensor *pred_buf, Tensor *grad_buf);

/* ── utilities ──────────────────────────────────────────────────── */
void nn_print_summary(const Network *net);
int  nn_total_params(const Network *net);

/* ── metrics ────────────────────────────────────────────────────── */
/*
 * nn_accuracy_binary:
 *   pred   [batch, 1]  sigmoid output (threshold 0.5)
 *   target [batch, 1]  ground truth 0 or 1
 *   Returns fraction correct in [0.0, 1.0]
 */
float nn_accuracy_binary(const Tensor *pred, const Tensor *target);

/*
 * nn_accuracy_multiclass:
 *   pred   [batch, classes]  softmax output
 *   target [batch, classes]  one-hot ground truth
 *   Returns fraction correct in [0.0, 1.0]
 */
float nn_accuracy_multiclass(const Tensor *pred, const Tensor *target);

/* ── weight I/O ─────────────────────────────────────────────────── */
int  nn_save(const Network *net, const char *path);
int  nn_load(Network *net, const char *path);

#endif /* NN_H */
