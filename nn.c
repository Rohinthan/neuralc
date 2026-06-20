#include "nn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

/* ── lifecycle ───────────────────────────────────────────────────── */

Network *nn_create(void) {
    Network *net = (Network *)calloc(1, sizeof(Network));
    return net;
}

void nn_add_layer(Network *net, DenseLayer *layer) {
    assert(net->num_layers < NN_MAX_LAYERS);
    net->layers[net->num_layers++] = layer;
}

void nn_free(Network *net) {
    if (!net) return;
    for (int i = 0; i < net->num_layers; i++) {
        dense_free(net->layers[i]);
        tensor_free(net->outputs[i]);
    }
    free(net);
}

/* ── forward ─────────────────────────────────────────────────────── */

void nn_forward(Network *net, const Tensor *input, Tensor *output) {
    const Tensor *cur = input;
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l   = net->layers[i];
        int batch       = cur->shape[0];
        int out_f       = l->out_features;

        /* allocate / reallocate layer output buffer if needed */
        if (!net->outputs[i] ||
            net->outputs[i]->shape[0] != batch ||
            net->outputs[i]->shape[1] != out_f) {
            tensor_free(net->outputs[i]);
            int sh[2] = {batch, out_f};
            net->outputs[i] = tensor_zeros(sh, 2);
        }

        dense_forward(l, cur, net->outputs[i]);
        cur = net->outputs[i];
    }
    /* copy final layer output to caller's buffer */
    assert(output->size == cur->size);
    tensor_copy_data(output, cur);
}

/* ── loss ────────────────────────────────────────────────────────── */

float nn_loss(LossType type, const Tensor *pred, const Tensor *target,
              Tensor *grad) {
    assert(pred->size == target->size);
    assert(grad->size == pred->size);
    size_t N = pred->size;
    float  loss = 0.0f;

    switch (type) {
    case LOSS_MSE:
        /* L = mean((pred - target)^2)
           dL/dpred = 2*(pred - target) / N               */
        for (size_t i = 0; i < N; i++) {
            float d = pred->data[i] - target->data[i];
            loss += d * d;
            grad->data[i] = 2.0f * d / (float)N;
        }
        loss /= (float)N;
        break;

    case LOSS_BINARY_CROSS:
        /* L = -mean(t*log(p) + (1-t)*log(1-p))
           dL/dp = -(t/p - (1-t)/(1-p)) / N               */
        for (size_t i = 0; i < N; i++) {
            float p = pred->data[i];
            float t = target->data[i];
            p = p < 1e-7f ? 1e-7f : (p > 1.0f-1e-7f ? 1.0f-1e-7f : p);
            loss -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
            grad->data[i] = (-(t / p) + (1.0f - t) / (1.0f - p)) / (float)N;
        }
        loss /= (float)N;
        break;

    case LOSS_CROSS_ENTROPY: {
        /* pred is already softmax output [batch, classes]
           target is one-hot or probability distribution
           L = -mean(sum_c t_c * log(p_c))
           dL/dp_c = -(t_c / p_c) / batch                 */
        int batch   = pred->shape[0];
        int classes = (int)(N / batch);
        for (int b = 0; b < batch; b++) {
            float row_loss = 0.0f;
            for (int c = 0; c < classes; c++) {
                float p = pred->data[b*classes + c];
                float t = target->data[b*classes + c];
                p = p < 1e-7f ? 1e-7f : p;
                row_loss -= t * logf(p);
                grad->data[b*classes + c] = -(t / p) / (float)batch;
            }
            loss += row_loss;
        }
        loss /= (float)batch;
        break;
    }
    }
    return loss;
}

/* ── backward ────────────────────────────────────────────────────── */

void nn_backward(Network *net, const Tensor *grad_out) {
    const Tensor *g = grad_out;
    for (int i = net->num_layers - 1; i >= 0; i--) {
        dense_backward(net->layers[i], g);
        g = net->layers[i]->dX;
    }
}

/* ── convenience training step ───────────────────────────────────── */

float nn_train_step(Network *net, const Tensor *input, const Tensor *target,
                    LossType loss_type, Tensor *pred_buf, Tensor *grad_buf) {
    nn_forward(net, input, pred_buf);
    float loss = nn_loss(loss_type, pred_buf, target, grad_buf);
    nn_backward(net, grad_buf);
    return loss;
}

/* ── utilities ───────────────────────────────────────────────────── */

void nn_print_summary(const Network *net) {
    printf("=== Network Summary ===\n");
    int total = 0;
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = net->layers[i];
        const char *act_names[] = {"None","ReLU","Sigmoid","Tanh","Softmax"};
        int p = dense_param_count(l);
        printf("  Layer %d: Dense(%d -> %d, %s)  params=%d\n",
               i, l->in_features, l->out_features,
               act_names[l->activation], p);
        total += p;
    }
    printf("  Total params: %d\n", total);
    printf("=======================\n");
}

int nn_total_params(const Network *net) {
    int t = 0;
    for (int i = 0; i < net->num_layers; i++)
        t += dense_param_count(net->layers[i]);
    return t;
}

/* ── weight I/O ──────────────────────────────────────────────────── */

int nn_save(const Network *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "nn_save: cannot open '%s'\n", path); return -1; }

    fwrite(&net->num_layers, sizeof(int), 1, f);
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = net->layers[i];
        fwrite(&l->in_features,  sizeof(int), 1, f);
        fwrite(&l->out_features, sizeof(int), 1, f);
        fwrite(&l->activation,   sizeof(int), 1, f);
        fwrite(l->W->data, sizeof(float), l->W->size, f);
        fwrite(l->b->data, sizeof(float), l->b->size, f);
    }
    fclose(f);
    return 0;
}

int nn_load(Network *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nn_load: cannot open '%s'\n", path); return -1; }

    int num_layers;
    fread(&num_layers, sizeof(int), 1, f);
    if (num_layers != net->num_layers) {
        fprintf(stderr, "nn_load: layer count mismatch (%d vs %d)\n",
                num_layers, net->num_layers);
        fclose(f); return -1;
    }
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = net->layers[i];
        int in_f, out_f, act;
        fread(&in_f,  sizeof(int), 1, f);
        fread(&out_f, sizeof(int), 1, f);
        fread(&act,   sizeof(int), 1, f);
        if (in_f != l->in_features || out_f != l->out_features) {
            fprintf(stderr, "nn_load: shape mismatch at layer %d\n", i);
            fclose(f); return -1;
        }
        fread(l->W->data, sizeof(float), l->W->size, f);
        fread(l->b->data, sizeof(float), l->b->size, f);
    }
    fclose(f);
    return 0;
}
