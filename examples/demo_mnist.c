/*
 * demo_mnist.c — neuralc MNIST handwritten digit recognition
 *
 * Network:  784 → 256 ReLU → Dropout → 128 ReLU → Dropout → 10 Softmax
 * Optimizer: Adam lr=0.001
 * Expected:  ~97% test accuracy
 *
 * Build:  make mnist_demo
 * Run:    ./mnist_demo
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "optimizer.h"
#include "dropout.h"
#include "mnist.h"

#define BATCH_SIZE  64
#define EPOCHS      20
#define LR          0.001f
#define DROP_RATE   0.3f

/* ── evaluate accuracy (no backward needed — safe to free early) ─── */
static float evaluate(Network *net,
                       DropoutLayer *d1, DropoutLayer *d2,
                       MNISTData *data) {
    dropout_eval(d1); dropout_eval(d2);
    int correct = 0;

    for (int s = 0; s < data->n; s += BATCH_SIZE) {
        int actual = BATCH_SIZE;
        if (s + actual > data->n) actual = data->n - s;

        int shX[2]={actual,784}, shY[2]={actual,10};
        int sh1[2]={actual,256}, sh2[2]={actual,128};

        Tensor *X   = tensor_zeros(shX,2);
        Tensor *Y   = tensor_zeros(shY,2);
        Tensor *h1  = tensor_zeros(sh1,2);
        Tensor *h1d = tensor_zeros(sh1,2);
        Tensor *h2  = tensor_zeros(sh2,2);
        Tensor *h2d = tensor_zeros(sh2,2);
        Tensor *out = tensor_zeros(shY,2);

        mnist_get_batch(data, s, actual, X, Y);

        dense_forward(net->layers[0], X,   h1);
        dropout_forward(d1, h1, h1d);
        dense_forward(net->layers[1], h1d, h2);
        dropout_forward(d2, h2, h2d);
        dense_forward(net->layers[2], h2d, out);

        int preds[BATCH_SIZE], trues[BATCH_SIZE];
        tensor_argmax_rows(out, preds);
        tensor_argmax_rows(Y,   trues);
        for (int i=0;i<actual;i++)
            if (preds[i]==trues[i]) correct++;

        /* safe to free all here — no backward needed */
        tensor_free(X);  tensor_free(Y);
        tensor_free(h1); tensor_free(h1d);
        tensor_free(h2); tensor_free(h2d);
        tensor_free(out);
    }
    return (float)correct / data->n * 100.0f;
}

/* ── main ────────────────────────────────────────────────────────── */
int main(void) {
    srand((unsigned)time(NULL));

    printf("╔══════════════════════════════════════════╗\n");
    printf("║   neuralc — MNIST Digit Recognition      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    /* load data */
    printf("Loading MNIST...\n");
    clock_t t0 = clock();

    MNISTData *train = mnist_load(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte", 0);
    MNISTData *test  = mnist_load(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",  0);

    if (!train || !test) {
        fprintf(stderr, "\nRun: bash mnist/download.sh\n");
        return 1;
    }
    printf("  Loaded in %.2fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    mnist_print_info(train, "train");
    mnist_print_info(test,  "test");

    printf("\nSample digit:\n");
    mnist_print_sample(train, 0);

    /* build network */
    printf("\nBuilding network...\n");
    Network *net = nn_create();
    nn_add_layer(net, dense_create(784, 256, ACT_RELU));
    nn_add_layer(net, dense_create(256, 128, ACT_RELU));
    nn_add_layer(net, dense_create(128,  10, ACT_SOFTMAX));
    nn_print_summary(net);

    DropoutLayer *d1  = dropout_create(DROP_RATE);
    DropoutLayer *d2  = dropout_create(DROP_RATE);
    Adam         *opt = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1e-4f);

    int n_batches = (train->n + BATCH_SIZE - 1) / BATCH_SIZE;

    printf("\n  %-6s %-12s %-12s %-10s %-8s\n",
           "Epoch","Train Loss","Train Acc","Test Acc","Time");
    printf("  %s\n","──────────────────────────────────────────────");

    float best_acc = 0.0f;

    for (int ep = 1; ep <= EPOCHS; ep++) {
        clock_t ep_t = clock();
        mnist_shuffle(train);
        dropout_train(d1); dropout_train(d2);

        float total_loss = 0.0f;
        int   correct    = 0;

        for (int b = 0; b < n_batches; b++) {
            int start  = b * BATCH_SIZE;
            int actual = BATCH_SIZE;
            if (start + actual > train->n) actual = train->n - start;

            /* ── allocate all tensors for this batch ── */
            int shX[2]={actual,784}, shY[2]={actual,10};
            int sh1[2]={actual,256}, sh2[2]={actual,128};

            Tensor *X    = tensor_zeros(shX,2);
            Tensor *Y    = tensor_zeros(shY,2);
            Tensor *h1   = tensor_zeros(sh1,2);
            Tensor *h1d  = tensor_zeros(sh1,2);
            Tensor *h2   = tensor_zeros(sh2,2);
            Tensor *h2d  = tensor_zeros(sh2,2);
            Tensor *out  = tensor_zeros(shY,2);
            Tensor *grad = tensor_zeros(shY,2);

            mnist_get_batch(train, start, actual, X, Y);

            /* ── FORWARD ──
             * Keep h1d and h2d alive! dense layers cache
             * their input pointer — needed in backward.   */
            dense_forward(net->layers[0], X,   h1);
            dropout_forward(d1, h1, h1d);
            dense_forward(net->layers[1], h1d, h2);
            dropout_forward(d2, h2, h2d);
            dense_forward(net->layers[2], h2d, out);

            /* ── LOSS ── */
            float loss = nn_loss(LOSS_CROSS_ENTROPY, out, Y, grad);
            total_loss += loss;

            int preds[BATCH_SIZE], trues[BATCH_SIZE];
            tensor_argmax_rows(out, preds);
            tensor_argmax_rows(Y,   trues);
            for (int i=0;i<actual;i++)
                if (preds[i]==trues[i]) correct++;

            /* ── BACKWARD ──
             * h1d, h2d still alive here — safe to backprop */
            dense_backward(net->layers[2], grad);

            Tensor *dh2 = tensor_zeros(sh2,2);
            dropout_backward(d2, net->layers[2]->dX, dh2);
            dense_backward(net->layers[1], dh2);
            tensor_free(dh2);

            Tensor *dh1 = tensor_zeros(sh1,2);
            dropout_backward(d1, net->layers[1]->dX, dh1);
            dense_backward(net->layers[0], dh1);
            tensor_free(dh1);

            /* ── UPDATE ── */
            adam_step(opt, net);

            /* ── FREE — safe now that backward is done ── */
            tensor_free(X);   tensor_free(Y);
            tensor_free(h1);  tensor_free(h1d);
            tensor_free(h2);  tensor_free(h2d);
            tensor_free(out); tensor_free(grad);
        }

        /* ── evaluate ── */
        float train_loss = total_loss / n_batches;
        float train_acc  = (float)correct / train->n * 100.0f;
        float test_acc   = evaluate(net, d1, d2, test);
        float ep_sec     = (float)(clock()-ep_t)/CLOCKS_PER_SEC;

        if (test_acc > best_acc) {
            best_acc = test_acc;
            nn_save(net, "mnist_best.bin");
        }

        dropout_train(d1); dropout_train(d2);

        printf("  %-6d %-12.4f %-11.1f%% %-9.1f%% %.1fs\n",
               ep, train_loss, train_acc, test_acc, ep_sec);
    }

    /* ── results ── */
    printf("\n  Best test accuracy: %.2f%%\n", best_acc);
    printf("  Model saved → mnist_best.bin\n");

    /* ── sample predictions ── */
    printf("\n  Sample Predictions:\n");
    printf("  %-6s %-10s %-10s\n","Index","Predicted","Actual");
    dropout_eval(d1); dropout_eval(d2);

    for (int i = 0; i < 10; i++) {
        int shx[2]={1,784}, shy[2]={1,10};
        int sha[2]={1,256},  shb[2]={1,128};

        Tensor *xi  = tensor_zeros(shx,2);
        Tensor *ha  = tensor_zeros(sha,2);
        Tensor *had = tensor_zeros(sha,2);
        Tensor *hb  = tensor_zeros(shb,2);
        Tensor *hbd = tensor_zeros(shb,2);
        Tensor *pi  = tensor_zeros(shy,2);

        memcpy(xi->data, test->images->data + i*784, 784*sizeof(float));

        dense_forward(net->layers[0], xi,  ha);
        dropout_forward(d1, ha, had);
        dense_forward(net->layers[1], had, hb);
        dropout_forward(d2, hb, hbd);
        dense_forward(net->layers[2], hbd, pi);

        int pred   = tensor_argmax(pi);
        int actual = (int)test->labels_raw->data[i];
        printf("  %-6d %-10d %-10d %s\n",
               i, pred, actual, pred==actual?"✓":"✗");

        tensor_free(xi);  tensor_free(ha);
        tensor_free(had); tensor_free(hb);
        tensor_free(hbd); tensor_free(pi);
    }

    printf("\n✓ neuralc trained on 60,000 real images in pure C!\n\n");

    dropout_free(d1); dropout_free(d2);
    adam_free(opt);
    nn_free(net);
    mnist_free(train);
    mnist_free(test);
    return 0;
}
