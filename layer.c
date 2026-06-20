#include "layer.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

/* ── helpers ─────────────────────────────────────────────────────── */

static Tensor *alloc2(int r, int c) {
    int sh[2] = {r, c};
    return tensor_zeros(sh, 2);
}
static Tensor *alloc1(int n) {
    int sh[1] = {n};
    return tensor_zeros(sh, 1);
}

/* ── lifecycle ───────────────────────────────────────────────────── */

DenseLayer *dense_create(int in_features, int out_features, Activation act) {
    DenseLayer *l = (DenseLayer *)calloc(1, sizeof(DenseLayer));
    if (!l) return NULL;
    l->in_features  = in_features;
    l->out_features = out_features;
    l->activation   = act;

    l->W  = alloc2(out_features, in_features);
    l->b  = alloc1(out_features);
    l->dW = alloc2(out_features, in_features);
    l->db = alloc1(out_features);

    dense_init_weights(l);
    return l;
}

void dense_free(DenseLayer *l) {
    if (!l) return;
    tensor_free(l->W);
    tensor_free(l->b);
    tensor_free(l->dW);
    tensor_free(l->db);
    tensor_free(l->z);
    tensor_free(l->a);
    tensor_free(l->dX);
    free(l);
}

void dense_init_weights(DenseLayer *l) {
    float scale;
    if (l->activation == ACT_RELU)
        scale = sqrtf(2.0f / l->in_features);          /* He   */
    else
        scale = sqrtf(1.0f / l->in_features);           /* Xavier */

    size_t n = l->W->size;
    /* use tensor_randn then scale */
    int sh[2] = {l->out_features, l->in_features};
    Tensor *tmp = tensor_randn(sh, 2);
    tensor_scale(tmp, scale, l->W);
    tensor_free(tmp);

    tensor_fill(l->b, 0.0f);
}

/* ── forward pass ────────────────────────────────────────────────── */
/*
 *  z = input @ W^T + b          [batch, out]
 *  a = activation(z)
 */
void dense_forward(DenseLayer *l, const Tensor *input, Tensor *output) {
    int batch = input->shape[0];
    int in_f  = input->shape[1];
    int out_f = l->out_features;
    assert(in_f == l->in_features);
    assert(output->shape[0] == batch && output->shape[1] == out_f);

    /* cache input pointer (not owned) */
    l->input = (Tensor *)input;   /* safe: we won't free it */

    /* allocate / resize z and a if needed */
    if (!l->z || l->z->shape[0] != batch) {
        tensor_free(l->z); tensor_free(l->a);
        l->z = alloc2(batch, out_f);
        l->a = alloc2(batch, out_f);
    }

    /* z = input @ W^T  (W is [out,in], so W^T is [in,out]) */
    /* We compute manually: z[b,o] = sum_i input[b,i]*W[o,i] + b[o] */
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_f; o++) {
            float acc = l->b->data[o];
            for (int i = 0; i < in_f; i++)
                acc += input->data[b*in_f + i] * l->W->data[o*in_f + i];
            l->z->data[b*out_f + o] = acc;
        }
    }

    /* activation */
    switch (l->activation) {
        case ACT_RELU:    tensor_relu(l->z, l->a);    break;
        case ACT_SIGMOID: tensor_sigmoid(l->z, l->a); break;
        case ACT_TANH:    tensor_tanh_t(l->z, l->a);  break;
        case ACT_SOFTMAX: tensor_softmax(l->z, l->a); break;
        default:          tensor_copy_data(l->a, l->z); break;
    }

    tensor_copy_data(output, l->a);
}

/* ── backward pass ───────────────────────────────────────────────── */
/*
 * grad_out: upstream gradient dL/da  [batch, out_f]
 *
 * 1. dL/dz  via activation derivative
 * 2. dL/dW  = dz^T @ input          [out_f, in_f]
 * 3. dL/db  = sum over batch of dz   [out_f]
 * 4. dL/dX  = dz @ W                [batch, in_f]
 */
void dense_backward(DenseLayer *l, const Tensor *grad_out) {
    int batch = l->input->shape[0];
    int in_f  = l->in_features;
    int out_f = l->out_features;

    /* allocate dz (same shape as z) */
    int sh2[2] = {batch, out_f};
    Tensor *dz = tensor_zeros(sh2, 2);

    /* activation gradient */
    switch (l->activation) {
        case ACT_RELU:
            tensor_relu_grad(l->z, grad_out, dz);
            break;
        case ACT_SIGMOID:
            tensor_sigmoid_grad(l->a, grad_out, dz);
            break;
        case ACT_TANH:
            tensor_tanh_grad(l->a, grad_out, dz);
            break;
        case ACT_SOFTMAX:
        case ACT_NONE:
        default:
            tensor_copy_data(dz, grad_out);
            break;
    }

    /* dW = dz^T @ input   [out_f, batch] x [batch, in_f] → [out_f, in_f] */
    tensor_fill(l->dW, 0.0f);
    for (int o = 0; o < out_f; o++)
        for (int b = 0; b < batch; b++) {
            float dzbo = dz->data[b*out_f + o];
            for (int i = 0; i < in_f; i++)
                l->dW->data[o*in_f + i] += dzbo * l->input->data[b*in_f + i];
        }
    /* average over batch */
    tensor_scale(l->dW, 1.0f / batch, l->dW);

    /* db = mean over batch of dz  [out_f] */
    tensor_fill(l->db, 0.0f);
    for (int b = 0; b < batch; b++)
        for (int o = 0; o < out_f; o++)
            l->db->data[o] += dz->data[b*out_f + o];
    tensor_scale(l->db, 1.0f / batch, l->db);

    /* dX = dz @ W   [batch, out_f] x [out_f, in_f] → [batch, in_f] */
    if (!l->dX || l->dX->shape[0] != batch) {
        tensor_free(l->dX);
        int shx[2] = {batch, in_f};
        l->dX = tensor_zeros(shx, 2);
    }
    tensor_fill(l->dX, 0.0f);
    for (int b = 0; b < batch; b++)
        for (int o = 0; o < out_f; o++) {
            float dzbo = dz->data[b*out_f + o];
            for (int i = 0; i < in_f; i++)
                l->dX->data[b*in_f + i] += dzbo * l->W->data[o*in_f + i];
        }

    tensor_free(dz);
}

int dense_param_count(const DenseLayer *l) {
    return l->in_features * l->out_features + l->out_features;
}
