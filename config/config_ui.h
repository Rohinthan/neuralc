#ifndef CONFIG_UI_H
#define CONFIG_UI_H

/*
 * config_ui.h — neuralc kernel-style configuration UI
 *
 * Terminal UI similar to Linux menuconfig.
 * Uses ANSI escape codes + termios — zero external dependencies.
 *
 * Usage:
 *   make config
 *   ./neuralc_config
 *
 * Generates: neuralc_config.h
 *
 * Navigation:
 *   ↑ ↓        — move between options
 *   Enter       — select / enter submenu
 *   Space       — toggle checkbox
 *   Y           — enable  [*]
 *   N           — disable [ ]
 *   S           — save config
 *   Q / Esc     — go back / quit
 *   H           — help
 */

/* ── item types ─────────────────────────────────────────────────── */
typedef enum {
    ITEM_TOGGLE,      /* [*] / [ ]  on/off checkbox          */
    ITEM_RADIO,       /* (*) / ( )  one of N choices         */
    ITEM_NUMBER,      /* integer value                        */
    ITEM_FLOAT,       /* float value                          */
    ITEM_SUBMENU,     /* ---> opens child menu               */
    ITEM_SEPARATOR,   /* ─────── visual divider              */
    ITEM_INFO,        /* read-only info line                  */
} ItemType;

#define MAX_ITEMS     64
#define MAX_CHILDREN  32
#define MAX_LABEL     80
#define MAX_MENUS     16

/* ── single config item ─────────────────────────────────────────── */
typedef struct ConfigItem {
    ItemType    type;
    char        label[MAX_LABEL];   /* display name                */
    char        key[64];            /* config key e.g. OMP_THREADS */
    char        help[256];          /* help text shown at bottom   */

    /* values */
    int         toggle;             /* 0 or 1 for TOGGLE           */
    int         radio_sel;          /* selected index for RADIO    */
    char        radio_opts[8][32];  /* option labels for RADIO     */
    int         radio_count;
    int         num_val;            /* integer value               */
    int         num_min, num_max;
    float       float_val;          /* float value                 */

    /* submenu */
    struct Menu *submenu;           /* pointer to child menu       */
} ConfigItem;

/* ── menu (a screen of items) ───────────────────────────────────── */
typedef struct Menu {
    char        title[MAX_LABEL];
    ConfigItem  items[MAX_ITEMS];
    int         count;
    int         cursor;             /* currently highlighted row   */
    int         scroll;             /* top visible row             */
} Menu;

/* ── global config state ────────────────────────────────────────── */
typedef struct {
    /* ── Performance ── */
    int   use_omp;          /* OpenMP multi-core            */
    int   omp_threads;      /* 0 = auto, else manual count  */
    int   omp_auto;         /* 1 = auto-detect cores        */
    int   use_gpu;          /* OpenCL GPU                   */
    int   use_blas;         /* BLAS integration             */

    /* ── Training defaults ── */
    int   batch_size;       /* default batch size           */
    float learning_rate;    /* default LR                   */
    int   optimizer;        /* 0=Adam 1=SGD 2=RMSProp       */
    float dropout_rate;     /* default dropout              */
    int   epochs;           /* default epochs               */
    float grad_clip;        /* gradient clip norm           */
    int   use_grad_clip;    /* enable gradient clipping     */

    /* ── Memory ── */
    int   allocator;        /* 0=malloc 1=pool              */
    int   pool_size_mb;     /* memory pool size in MB       */

    /* ── Debug ── */
    int   debug_mode;       /* verbose logging              */
    int   check_nan;        /* check for NaN in tensors     */
    int   profile;          /* print timing info            */

    /* ── Build ── */
    int   opt_level;        /* 0=O0 1=O1 2=O2 3=O3         */
    int   enable_avx;       /* AVX/SIMD instructions        */
    int   enable_lto;       /* link-time optimization       */
} NeuralcConfig;

/* ── API ────────────────────────────────────────────────────────── */

/* Run the interactive config UI. Returns 0 if saved, 1 if quit. */
int  config_ui_run(NeuralcConfig *cfg);

/* Load config from neuralc_config.h (if exists) */
int  config_load(NeuralcConfig *cfg, const char *path);

/* Save config to neuralc_config.h */
int  config_save(const NeuralcConfig *cfg, const char *path);

/* Print current config summary */
void config_print(const NeuralcConfig *cfg);

/* Initialize config with sensible defaults */
void config_defaults(NeuralcConfig *cfg);

#endif /* CONFIG_UI_H */
