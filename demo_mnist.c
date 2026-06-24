/*
 * demo_mnist.c — neuralc MNIST handwritten digit recognition
 *
 * Architecture:
 *   Input  784 (28x28 flattened)
 *   Dense  256  ReLU
 *   Dropout 0.3
 *   Dense  128  ReLU
 *   Dropout 0.3
 *   Dense   10  Softmax
 *
 * Optimizer: Adam lr=0.001
 * Loss:      Cross-Entropy
 * Epochs:    20
 * Batch:     64
 *
 * Expected accuracy: ~97% on test set
 *
 * Setup:
 *   bash mnist/download.sh
 *   make mnist_demo
 *   ./mnist_demo
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

/* ── config ─────────────────────────────────────────────────────── */
#define BATCH_SIZE   64
#define EPOCHS       20
#define LR           0.001f
#define DROP_RATE    0.3f
#define TRAIN_LIMIT  0      /* 0 = all 60000, set e.g. 10000 to test fast */
#define TEST_LIMIT   0      /* 0 = all 10000 */

/* ── timing ─────────────────────────────────────────────────────── */
static double now_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

/* ── evaluate accuracy on a dataset ─────────────────────────────── */
static float evaluate(Network *net, MNISTData *data,
                       Tensor *X_buf, Tensor *Y_buf) {
    int correct = 0;
    int total   = data->n;
    int bs      = BATCH_SIZE;

    for (int start = 0; start < total; start += bs) {
        int actual = bs;
        if (start + actual > total) actual = total - start;

        /* resize buffers if last batch is smaller */
        int shX[2] = {actual, 784};
        int shY[2] = {actual, 10};
        Tensor *Xb = tensor_zeros(shX, 2);
        Tensor *Yb = tensor_zeros(shY, 2);
        Tensor *Pb = tensor_zeros(shY, 2);

        mnist_get_batch(data, start, actual, Xb, Yb);
        nn_forward(net, Xb, Pb);

        /* count correct predictions */
        int preds[BATCH_SIZE];
        int trues[BATCH_SIZE];
        tensor_argmax_rows(Pb, preds);
        tensor_argmax_rows(Yb, trues);
        for (int i = 0; i < actual; i++)
            if (preds[i] == trues[i]) correct++;

        tensor_free(Xb); tensor_free(Yb); tensor_free(Pb);
    }
    return (float)correct / (float)total * 100.0f;
}

/* ── main ────────────────────────────────────────────────────────── */

int main(void) {
    srand((unsigned)time(NULL));

    printf("╔══════════════════════════════════════════╗\n");
    printf("║   neuralc — MNIST Digit Recognition      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    /* ── load data ── */
    printf("Loading MNIST...\n");
    double t0 = now_sec();

    MNISTData *train = mnist_load(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte",
        TRAIN_LIMIT);

    MNISTData *test = mnist_load(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",
        TEST_LIMIT);

    if (!train || !test) {
        fprintf(stderr,
            "\nMNIST files not found!\n"
            "Run this first:\n"
            "  bash mnist/download.sh\n\n");
        return 1;
    }

    printf("  Loaded in %.2fs\n", now_sec() - t0);
    mnist_print_info(train, "train");
    mnist_print_info(test,  "test");

    /* ── show a sample ── */
    printf("\nSample digit from training set:\n");
    mnist_print_sample(train, 0);

    /* ── build network ── */
    printf("\nBuilding network...\n");
    Network *net = nn_create();
    nn_add_layer(net, dense_create(784, 256, ACT_RELU));
    nn_add_layer(net, dense_create(256, 128, ACT_RELU));
    nn_add_layer(net, dense_create(128,  10, ACT_SOFTMAX));
    nn_print_summary(net);

    /* dropout layers (applied manually between dense layers) */
    DropoutLayer *drop1 = dropout_create(DROP_RATE);
    DropoutLayer *drop2 = dropout_create(DROP_RATE);

    /* ── optimizer ── */
    Adam *opt = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1e-4f);

    /* ── training buffers ── */
    int shX[2] = {BATCH_SIZE, 784};
    int shY[2] = {BATCH_SIZE, 10};
    Tensor *X_batch = tensor_zeros(shX, 2);
    Tensor *Y_batch = tensor_zeros(shY, 2);
    Tensor *pred    = tensor_zeros(shY, 2);
    Tensor *grad    = tensor_zeros(shY, 2);

    int n_train   = train->n;
    int n_batches = (n_train + BATCH_SIZE - 1) / BATCH_SIZE;

    printf("\n  Epochs:%d  BatchSize:%d  LR:%.4f  Dropout:%.1f\n\n",
           EPOCHS, BATCH_SIZE, LR, DROP_RATE);
    printf("  %-6s %-12s %-12s %-10s %-10s\n",
           "Epoch", "Train Loss", "Train Acc", "Test Acc", "Time");
    printf("  %s\n", "──────────────────────────────────────────────");

    float best_test_acc = 0.0f;

    /* ── training loop ── */
    for (int ep = 1; ep <= EPOCHS; ep++) {
        double ep_start = now_sec();

        /* shuffle each epoch */
        mnist_shuffle(train);

        float total_loss = 0.0f;
        int   correct    = 0;

        /* set dropout to training mode */
        dropout_train(drop1);
        dropout_train(drop2);

        for (int b = 0; b < n_batches; b++) {
            int start  = b * BATCH_SIZE;
            int actual = BATCH_SIZE;
            if (start + actual > n_train) actual = n_train - start;
            if (actual <= 0) break;

            /* get batch */
            mnist_get_batch(train, start, actual, X_batch, Y_batch);

            /*
             * Manual forward with dropout:
             * Layer 0 (784→256 ReLU) → dropout1
             * Layer 1 (256→128 ReLU) → dropout2
             * Layer 2 (128→10 Softmax)
             */
            int sh0[2]={actual,784}, sh1[2]={actual,256};
            int sh2[2]={actual,128}, sh3[2]={actual,10};

            Tensor *x0  = tensor_zeros(sh0,2); memcpy(x0->data,X_batch->data,actual*784*sizeof(float));
            Tensor *h1  = tensor_zeros(sh1,2);
            Tensor *h1d = tensor_zeros(sh1,2);
            Tensor *h2  = tensor_zeros(sh2,2);
            Tensor *h2d = tensor_zeros(sh2,2);
            Tensor *out = tensor_zeros(sh3,2);

            dense_forward(net->layers[0], x0, h1);
            dropout_forward(drop1, h1, h1d);
            dense_forward(net->layers[1], h1d, h2);
            dropout_forward(drop2, h2, h2d);
            dense_forward(net->layers[2], h2d, out);

            /* loss */
            float loss = nn_loss(LOSS_CROSS_ENTROPY,
                                 out, Y_batch, grad);
            total_loss += loss;

            /* accuracy */
            int preds_arr[BATCH_SIZE], trues_arr[BATCH_SIZE];
            tensor_argmax_rows(out, preds_arr);
            tensor_argmax_rows(Y_batch, trues_arr);
            for (int i=0;i<actual;i++)
                if (preds_arr[i]==trues_arr[i]) correct++;

            /* backward */
            dense_backward(net->layers[2], grad);

            /* dropout2 backward */
            Tensor *dh2d = tensor_zeros(sh2,2);
            dropout_backward(drop2, net->layers[2]->dX, dh2d);
            dense_backward(net->layers[1], dh2d);

            /* dropout1 backward */
            Tensor *dh1d = tensor_zeros(sh1,2);
            dropout_backward(drop1, net->layers[1]->dX, dh1d);
            dense_backward(net->layers[0], dh1d);

            /* update */
            adam_step(opt, net);

            tensor_free(x0); tensor_free(h1); tensor_free(h1d);
            tensor_free(h2); tensor_free(h2d); tensor_free(out);
            tensor_free(dh2d); tensor_free(dh1d);
        }

        /* ── evaluation ── */
        dropout_eval(drop1);
        dropout_eval(drop2);

        float train_loss = total_loss / n_batches;
        float train_acc  = (float)correct / n_train * 100.0f;
        float test_acc   = evaluate(net, test, X_batch, Y_batch);
        double ep_time   = now_sec() - ep_start;

        if (test_acc > best_test_acc) best_test_acc = test_acc;

        printf("  %-6d %-12.4f %-12.1f%% %-10.1f%% %.1fs\n",
               ep, train_loss, train_acc, test_acc, ep_time);

        /* save best model */
        if (test_acc >= best_test_acc)
            nn_save(net, "mnist_best.bin");
    }

    /* ── final results ── */
    printf("\n  %s\n", "──────────────────────────────────────────────");
    printf("  Best test accuracy: %.2f%%\n", best_test_acc);
    printf("  Model saved: mnist_best.bin\n");

    /* ── show predictions on 10 test samples ── */
    printf("\n  Sample predictions:\n");
    printf("  %-8s %-10s %-10s\n", "Index", "Predicted", "Actual");
    dropout_eval(drop1); dropout_eval(drop2);

    for (int i = 0; i < 10; i++) {
        int sh1b[2]={1,784}, sh2b[2]={1,10};
        Tensor *xi = tensor_zeros(sh1b,2);
        Tensor *pi = tensor_zeros(sh2b,2);
        memcpy(xi->data, test->images->data + i*784, 784*sizeof(float));

        /* manual forward */
        int sha[2]={1,256}, shb[2]={1,128};
        Tensor *ha  = tensor_zeros(sha,2);
        Tensor *had = tensor_zeros(sha,2);
        Tensor *hb  = tensor_zeros(shb,2);
        Tensor *hbd = tensor_zeros(shb,2);
        dense_forward(net->layers[0],xi,ha);
        dropout_forward(drop1,ha,had);
        dense_forward(net->layers[1],had,hb);
        dropout_forward(drop2,hb,hbd);
        dense_forward(net->layers[2],hbd,pi);

        int predicted = tensor_argmax(pi);
        int actual    = (int)test->labels_raw->data[i];
        char mark     = (predicted == actual) ? ' ' : '!';

        printf("  %-8d %-10d %-10d %c\n",
               i, predicted, actual, mark);

        tensor_free(xi); tensor_free(pi);
        tensor_free(ha); tensor_free(had);
        tensor_free(hb); tensor_free(hbd);
    }

    printf("\n✓ MNIST demo complete!\n\n");

    /* ── cleanup ── */
    tensor_free(X_batch); tensor_free(Y_batch);
    tensor_free(pred);    tensor_free(grad);
    dropout_free(drop1);  dropout_free(drop2);
    adam_free(opt);
    nn_free(net);
    mnist_free(train);
    mnist_free(test);

    return 0;
}
