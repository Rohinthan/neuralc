/*
 * config_ui.c — neuralc kernel-style configuration UI
 *
 * Terminal UI using ANSI escape codes + POSIX termios.
 * No external dependencies required.
 *
 * Looks and feels like Linux menuconfig.
 */

#include "config_ui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <math.h>
#include <time.h>

/* ── ANSI color codes ────────────────────────────────────────────── */
#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define DIM         "\033[2m"

/* foreground */
#define FG_BLACK    "\033[30m"
#define FG_RED      "\033[31m"
#define FG_GREEN    "\033[32m"
#define FG_YELLOW   "\033[33m"
#define FG_BLUE     "\033[34m"
#define FG_MAGENTA  "\033[35m"
#define FG_CYAN     "\033[36m"
#define FG_WHITE    "\033[37m"

/* background */
#define BG_BLACK    "\033[40m"
#define BG_RED      "\033[41m"
#define BG_GREEN    "\033[42m"
#define BG_YELLOW   "\033[43m"
#define BG_BLUE     "\033[44m"
#define BG_MAGENTA  "\033[45m"
#define BG_CYAN     "\033[46m"
#define BG_WHITE    "\033[47m"

/* cursor control */
#define CLEAR_SCREEN "\033[2J\033[H"
#define CURSOR_HIDE  "\033[?25l"
#define CURSOR_SHOW  "\033[?25h"
#define CURSOR_POS(r,c) printf("\033[%d;%dH", (r), (c))

/* ── terminal state ──────────────────────────────────────────────── */
static struct termios old_term;
static int term_rows = 24;
static int term_cols = 80;

static void term_raw(void) {
    struct termios raw;
    tcgetattr(STDIN_FILENO, &old_term);
    raw = old_term;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(CURSOR_HIDE);
}

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_term);
    printf(CURSOR_SHOW);
    printf(RESET);
}

static void term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        term_rows = ws.ws_row;
        term_cols = ws.ws_col;
    }
}

/* ── keyboard input ──────────────────────────────────────────────── */
#define KEY_UP      1000
#define KEY_DOWN    1001
#define KEY_LEFT    1002
#define KEY_RIGHT   1003
#define KEY_ENTER   '\n'
#define KEY_ESC     27
#define KEY_SPACE   ' '

static int read_key(void) {
    unsigned char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return -1;
    if (c == KEY_ESC) {
        unsigned char seq[3];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return KEY_ESC;
        if (seq[0] == '[') {
            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        return KEY_ESC;
    }
    return (int)c;
}

/* ── drawing helpers ─────────────────────────────────────────────── */





static void draw_border_box(int row, int col, int h, int w,
                             const char *title) {
    /* top border */
    CURSOR_POS(row, col);
    printf(BG_BLACK FG_CYAN "┌");
    if (title && strlen(title) > 0) {
        printf("─ " FG_WHITE BOLD "%s" RESET BG_BLACK FG_CYAN " ", title);
        int used = (int)strlen(title) + 4;
        for (int i = used; i < w - 1; i++) printf("─");
    } else {
        for (int i = 1; i < w - 1; i++) printf("─");
    }
    printf("┐");

    /* sides */
    for (int r = 1; r < h - 1; r++) {
        CURSOR_POS(row + r, col);
        printf(BG_BLACK FG_CYAN "│");
        printf(BG_BLACK FG_WHITE);
        for (int i = 1; i < w - 1; i++) printf(" ");
        printf(FG_CYAN "│");
    }

    /* bottom */
    CURSOR_POS(row + h - 1, col);
    printf(BG_BLACK FG_CYAN "└");
    for (int i = 1; i < w - 1; i++) printf("─");
    printf("┘" RESET);
}

/* ── item rendering ──────────────────────────────────────────────── */

static void render_item(const ConfigItem *it, int row, int col,
                         int width, int selected) {
    CURSOR_POS(row, col);

    if (selected)
        printf(BG_WHITE FG_BLACK BOLD);
    else
        printf(BG_BLACK FG_WHITE);

    /* clear line */
    for (int i = 0; i < width; i++) printf(" ");
    CURSOR_POS(row, col + 1);

    switch (it->type) {
    case ITEM_SEPARATOR:
        printf(BG_BLACK FG_CYAN DIM);
        printf("  ─────────────────────────────────────");
        break;

    case ITEM_INFO:
        if (selected) printf(BG_WHITE FG_BLACK);
        else          printf(BG_BLACK FG_YELLOW);
        printf("  %s", it->label);
        break;

    case ITEM_TOGGLE:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  [%s] %s",
               it->toggle ? "*" : " ",
               it->label);
        break;

    case ITEM_RADIO:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%s) %s  ",
               it->radio_opts[it->radio_sel],
               it->label);
        if (selected) {
            printf(BG_WHITE FG_BLUE);
            printf("< ");
            for (int i = 0; i < it->radio_count; i++) {
                if (i == it->radio_sel)
                    printf(BOLD "(%s)" RESET BG_WHITE FG_BLUE,
                           it->radio_opts[i]);
                else
                    printf(" %s ", it->radio_opts[i]);
                if (i < it->radio_count - 1) printf(" | ");
            }
            printf(" >");
        }
        break;

    case ITEM_NUMBER:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%d) %s", it->num_val, it->label);
        if (selected)
            printf(FG_CYAN "  [← → to change, min:%d max:%d]",
                   it->num_min, it->num_max);
        break;

    case ITEM_FLOAT:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%.4f) %s", it->float_val, it->label);
        if (selected)
            printf(FG_CYAN "  [← to halve, → to double]");
        break;

    case ITEM_SUBMENU:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_CYAN);
        printf("  %s  --->", it->label);
        break;
    }

    printf(RESET);
}

/* ── help bar ────────────────────────────────────────────────────── */

static void draw_help_bar(int row, const char *help_text) {
    CURSOR_POS(row, 1);
    printf(BG_BLACK FG_CYAN);
    for (int i = 0; i < term_cols; i++) printf("─");

    CURSOR_POS(row + 1, 1);
    printf(BG_BLACK FG_YELLOW);
    printf("  %s", help_text ? help_text :
           "Use arrows to navigate, Space/Enter to toggle, S to save, Q to quit");
    for (int i = 0; i < term_cols - 2; i++) printf(" ");

    CURSOR_POS(row + 2, 1);
    printf(BG_BLUE FG_WHITE BOLD);
    /* bottom button bar */
    int bw = term_cols / 5;
    char btns[5][20] = {"<Select>","<Exit>","<Help>","<Save>","<Load>"};
    for (int i = 0; i < 5; i++) {
        printf(BG_WHITE FG_BLACK " %-*s" RESET BG_BLUE, bw-1, btns[i]);
    }
    printf(RESET);
}

/* ── main menu renderer ──────────────────────────────────────────── */

#define MENU_TOP    4
#define MENU_MARGIN 4

static void draw_menu(Menu *m, const char *breadcrumb) {
    term_size();
    printf(CLEAR_SCREEN);

    /* title bar */
    CURSOR_POS(1, 1);
    printf(BG_BLUE FG_WHITE BOLD);
    for (int i = 0; i < term_cols; i++) printf(" ");
    CURSOR_POS(1, 1);
    printf("  neuralc Configuration");
    CURSOR_POS(1, term_cols - 20);
    printf("neuralc v0.1  ");
    printf(RESET);

    /* subtitle / breadcrumb */
    CURSOR_POS(2, 1);
    printf(BG_BLACK FG_CYAN);
    for (int i = 0; i < term_cols; i++) printf(" ");
    CURSOR_POS(2, 3);
    printf(FG_YELLOW BOLD "► " FG_WHITE "%s", breadcrumb);
    printf(RESET);

    /* menu box */
    int box_h = term_rows - MENU_TOP - 4;
    int box_w = term_cols - MENU_MARGIN * 2;
    draw_border_box(MENU_TOP, MENU_MARGIN, box_h, box_w, m->title);

    /* visible range */
    int visible = box_h - 2;
    if (m->cursor < m->scroll) m->scroll = m->cursor;
    if (m->cursor >= m->scroll + visible) m->scroll = m->cursor - visible + 1;
    if (m->scroll < 0) m->scroll = 0;

    for (int i = 0; i < visible && (m->scroll + i) < m->count; i++) {
        int idx = m->scroll + i;
        render_item(&m->items[idx],
                    MENU_TOP + 1 + i,
                    MENU_MARGIN + 1,
                    box_w - 2,
                    idx == m->cursor);
    }

    /* scroll indicator */
    if (m->count > visible) {
        CURSOR_POS(MENU_TOP + 1, MENU_MARGIN + box_w - 3);
        printf(BG_BLACK FG_CYAN "%d/%d" RESET, m->cursor + 1, m->count);
    }

    /* help bar */
    const char *help = m->items[m->cursor].help;
    draw_help_bar(term_rows - 2, help[0] ? help : NULL);

    fflush(stdout);
}

/* ── menu navigation ─────────────────────────────────────────────── */

static int next_selectable(Menu *m, int from, int dir) {
    int i = from + dir;
    while (i >= 0 && i < m->count) {
        if (m->items[i].type != ITEM_SEPARATOR &&
            m->items[i].type != ITEM_INFO)
            return i;
        i += dir;
    }
    return from;
}

/* ── popup dialogs ───────────────────────────────────────────────── */

static void popup_message(const char *title, const char *msg) {
    int w = 50, h = 7;
    int r = (term_rows - h) / 2;
    int c = (term_cols - w) / 2;
    draw_border_box(r, c, h, w, title);
    CURSOR_POS(r + 2, c + 3);
    printf(BG_BLACK FG_WHITE "  %s", msg);
    CURSOR_POS(r + 4, c + w/2 - 4);
    printf(BG_WHITE FG_BLACK BOLD " <  OK  > " RESET);
    fflush(stdout);
    read_key();
}



/* ── run a menu ──────────────────────────────────────────────────── */

static int run_menu(Menu *m, const char *breadcrumb);  /* forward decl */

static int handle_item(Menu *m, int idx, const char *breadcrumb) {
    ConfigItem *it = &m->items[idx];
    switch (it->type) {
    case ITEM_TOGGLE:
        it->toggle = !it->toggle;
        break;
    case ITEM_RADIO:
        it->radio_sel = (it->radio_sel + 1) % it->radio_count;
        break;
    case ITEM_NUMBER:
        if (it->num_val < it->num_max)
            it->num_val++;
        break;
    case ITEM_FLOAT:
        it->float_val *= 2.0f;
        break;
    case ITEM_SUBMENU:
        if (it->submenu) {
            char bc[256];
            snprintf(bc, sizeof(bc), "%s > %s", breadcrumb, it->label);
            return run_menu(it->submenu, bc);
        }
        break;
    default:
        break;
    }
    return 0;
}

static int run_menu(Menu *m, const char *breadcrumb) {
    while (1) {
        draw_menu(m, breadcrumb);
        int k = read_key();
        switch (k) {
        case KEY_UP:
            m->cursor = next_selectable(m, m->cursor, -1);
            break;
        case KEY_DOWN:
            m->cursor = next_selectable(m, m->cursor, +1);
            break;
        case KEY_LEFT: {
            ConfigItem *it = &m->items[m->cursor];
            if (it->type == ITEM_NUMBER && it->num_val > it->num_min)
                it->num_val--;
            else if (it->type == ITEM_FLOAT)
                it->float_val /= 2.0f;
            else if (it->type == ITEM_RADIO)
                it->radio_sel = (it->radio_sel + it->radio_count - 1)
                                 % it->radio_count;
            break;
        }
        case KEY_RIGHT: {
            ConfigItem *it = &m->items[m->cursor];
            if (it->type == ITEM_NUMBER && it->num_val < it->num_max)
                it->num_val++;
            else if (it->type == ITEM_FLOAT)
                it->float_val *= 2.0f;
            else if (it->type == ITEM_RADIO)
                it->radio_sel = (it->radio_sel + 1) % it->radio_count;
            break;
        }
        case KEY_ENTER:
        case KEY_SPACE:
            if (handle_item(m, m->cursor, breadcrumb) == 1)
                return 1;   /* save signal */
            break;
        case 'y': case 'Y':
            if (m->items[m->cursor].type == ITEM_TOGGLE)
                m->items[m->cursor].toggle = 1;
            break;
        case 'n': case 'N':
            if (m->items[m->cursor].type == ITEM_TOGGLE)
                m->items[m->cursor].toggle = 0;
            break;
        case 's': case 'S':
            return 1;   /* save */
        case 'q': case 'Q':
        case KEY_ESC:
            return 0;   /* back/quit */
        case 'h': case 'H': case '?':
            popup_message("Help",
                "↑↓ Navigate  Space/Enter Toggle  "
                "←→ Change value  S Save  Q Quit");
            break;
        }
    }
}

/* ── menu builder helpers ────────────────────────────────────────── */

static void add_sep(Menu *m, const char *label) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type = ITEM_SEPARATOR;
    snprintf(it->label, MAX_LABEL, "%s", label ? label : "");
}

static void add_toggle(Menu *m, const char *label, const char *key,
                        int val, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type   = ITEM_TOGGLE;
    it->toggle = val;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_radio(Menu *m, const char *label, const char *key,
                       int sel, int n, const char **opts,
                       const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type      = ITEM_RADIO;
    it->radio_sel = sel;
    it->radio_count = n;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    for (int i = 0; i < n && i < 8; i++)
        snprintf(it->radio_opts[i], 32, "%s", opts[i]);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_number(Menu *m, const char *label, const char *key,
                        int val, int mn, int mx, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type    = ITEM_NUMBER;
    it->num_val = val;
    it->num_min = mn;
    it->num_max = mx;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_float(Menu *m, const char *label, const char *key,
                       float val, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type      = ITEM_FLOAT;
    it->float_val = val;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_submenu(Menu *m, const char *label, Menu *sub,
                         const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type    = ITEM_SUBMENU;
    it->submenu = sub;
    snprintf(it->label, MAX_LABEL, "%s", label);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_info(Menu *m, const char *text) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type = ITEM_INFO;
    snprintf(it->label, MAX_LABEL, "%s", text);
}

/* ── build neuralc menu tree ─────────────────────────────────────── */

static Menu perf_menu, train_menu, mem_menu, debug_menu, build_menu;
static Menu root_menu;

static void build_menus(NeuralcConfig *cfg) {
    memset(&perf_menu,  0, sizeof(Menu));
    memset(&train_menu, 0, sizeof(Menu));
    memset(&mem_menu,   0, sizeof(Menu));
    memset(&debug_menu, 0, sizeof(Menu));
    memset(&build_menu, 0, sizeof(Menu));
    memset(&root_menu,  0, sizeof(Menu));

    /* ── Performance submenu ── */
    snprintf(perf_menu.title, MAX_LABEL, "Performance Settings");
    add_info(&perf_menu, "Configure CPU/GPU acceleration options");
    add_sep(&perf_menu, NULL);
    add_toggle(&perf_menu, "Enable OpenMP multi-core", "USE_OMP",
               cfg->use_omp,
               "Use multiple CPU cores in parallel for tensor operations");
    {
        const char *ta[] = {"Auto","Manual"};
        add_radio(&perf_menu, "Thread Allocation", "OMP_THREAD_MODE",
                  cfg->omp_auto ? 0 : 1, 2, ta,
                  "Auto = detect all CPU cores, Manual = set count below");
    }
    add_number(&perf_menu, "Thread Count (Manual mode)", "OMP_THREADS",
               cfg->omp_threads, 1, 256,
               "Number of threads for OpenMP (only used in Manual mode)");
    add_sep(&perf_menu, NULL);
    add_toggle(&perf_menu, "Enable OpenCL GPU", "USE_GPU",
               cfg->use_gpu,
               "Use GPU via OpenCL for tensor operations (requires OpenCL)");
    add_toggle(&perf_menu, "Enable BLAS integration", "USE_BLAS",
               cfg->use_blas,
               "Use OpenBLAS for faster matrix multiply (install: apt install libopenblas-dev)");

    /* ── Training submenu ── */
    snprintf(train_menu.title, MAX_LABEL, "Training Defaults");
    add_info(&train_menu, "Default hyperparameters for training");
    add_sep(&train_menu, NULL);
    add_number(&train_menu, "Default Batch Size", "DEFAULT_BATCH",
               cfg->batch_size, 1, 4096,
               "Samples per gradient update. Larger = faster but more memory");
    add_float(&train_menu, "Default Learning Rate", "DEFAULT_LR",
              cfg->learning_rate,
              "Step size for gradient descent. Adam typical: 0.001");
    add_number(&train_menu, "Default Epochs", "DEFAULT_EPOCHS",
               cfg->epochs, 1, 10000,
               "Number of full passes over the training dataset");
    add_sep(&train_menu, NULL);
    {
        const char *opts[] = {"Adam","SGD","RMSProp"};
        add_radio(&train_menu, "Default Optimizer", "DEFAULT_OPT",
                  cfg->optimizer, 3, opts,
                  "Adam is best for most tasks. SGD needs tuned LR. RMSProp good for RNN");
    }
    add_float(&train_menu, "Default Dropout Rate", "DEFAULT_DROPOUT",
              cfg->dropout_rate,
              "Fraction of neurons to drop during training. 0.0 = disabled");
    add_sep(&train_menu, NULL);
    add_toggle(&train_menu, "Enable Gradient Clipping", "USE_GRAD_CLIP",
               cfg->use_grad_clip,
               "Prevent exploding gradients — essential for RNN/LSTM");
    add_float(&train_menu, "Gradient Clip Norm", "GRAD_CLIP_NORM",
              cfg->grad_clip,
              "Maximum gradient L2 norm. Typical: 1.0 for RNN, 5.0 for Dense");

    /* ── Memory submenu ── */
    snprintf(mem_menu.title, MAX_LABEL, "Memory Settings");
    add_info(&mem_menu, "Control memory allocation strategy");
    add_sep(&mem_menu, NULL);
    {
        const char *ma[] = {"malloc","Pool"};
        add_radio(&mem_menu, "Memory Allocator", "ALLOCATOR",
                  cfg->allocator, 2, ma,
                  "malloc = standard, Pool = pre-allocate for speed");
    }
    add_number(&mem_menu, "Memory Pool Size (MB)", "POOL_SIZE_MB",
               cfg->pool_size_mb, 64, 16384,
               "Size of pre-allocated memory pool (Pool allocator only)");

    /* ── Debug submenu ── */
    snprintf(debug_menu.title, MAX_LABEL, "Debug & Profiling");
    add_info(&debug_menu, "Enable diagnostic and profiling features");
    add_sep(&debug_menu, NULL);
    add_toggle(&debug_menu, "Debug Mode (verbose logging)", "DEBUG_MODE",
               cfg->debug_mode,
               "Print detailed logs during training. Slows down execution");
    add_toggle(&debug_menu, "Check for NaN in tensors", "CHECK_NAN",
               cfg->check_nan,
               "Detect NaN/Inf values in tensors. Useful for debugging instability");
    add_toggle(&debug_menu, "Profiling (print timing)", "PROFILE",
               cfg->profile,
               "Print timing information for each operation");

    /* ── Build submenu ── */
    snprintf(build_menu.title, MAX_LABEL, "Build Options");
    add_info(&build_menu, "Compiler and build configuration");
    add_sep(&build_menu, NULL);
    {
        const char *ol[] = {"-O0","-O1","-O2","-O3"};
        add_radio(&build_menu, "Optimization Level", "OPT_LEVEL",
                  cfg->opt_level, 4, ol,
                  "-O2 is default. -O3 is faster but may cause issues. -O0 for debugging");
    }
    add_toggle(&build_menu, "Enable AVX/SIMD instructions", "ENABLE_AVX",
               cfg->enable_avx,
               "Use CPU vector instructions for faster math. Requires modern CPU");
    add_toggle(&build_menu, "Enable Link-Time Optimization", "ENABLE_LTO",
               cfg->enable_lto,
               "Optimize across compilation units. Slower build, faster runtime");

    /* ── Root menu ── */
    snprintf(root_menu.title, MAX_LABEL,
             "neuralc v0.1 Configuration");
    add_info(&root_menu,
             "Arrow keys navigate. Space/Enter toggle. S save. Q quit.");
    add_sep(&root_menu, NULL);
    add_submenu(&root_menu, "Performance Settings",   &perf_menu,
                "OpenMP threads, GPU, BLAS");
    add_submenu(&root_menu, "Training Defaults",      &train_menu,
                "Batch size, LR, optimizer, gradient clipping");
    add_submenu(&root_menu, "Memory Settings",        &mem_menu,
                "Allocator and pool size");
    add_submenu(&root_menu, "Debug & Profiling",      &debug_menu,
                "Logging, NaN check, timing");
    add_submenu(&root_menu, "Build Options",          &build_menu,
                "Compiler flags, AVX, LTO");
    add_sep(&root_menu, NULL);
    add_info(&root_menu, "Press S to save  |  Q to quit without saving");
}

/* ── read back values after UI ───────────────────────────────────── */

static void collect_config(NeuralcConfig *cfg) {
    /* performance */
    cfg->use_omp      = perf_menu.items[2].toggle;
    cfg->omp_auto     = (perf_menu.items[3].radio_sel == 0);
    cfg->omp_threads  = perf_menu.items[4].num_val;
    cfg->use_gpu      = perf_menu.items[6].toggle;
    cfg->use_blas     = perf_menu.items[7].toggle;

    /* training */
    cfg->batch_size   = train_menu.items[2].num_val;
    cfg->learning_rate= train_menu.items[3].float_val;
    cfg->epochs       = train_menu.items[4].num_val;
    cfg->optimizer    = train_menu.items[6].radio_sel;
    cfg->dropout_rate = train_menu.items[7].float_val;
    cfg->use_grad_clip= train_menu.items[9].toggle;
    cfg->grad_clip    = train_menu.items[10].float_val;

    /* memory */
    cfg->allocator    = mem_menu.items[2].radio_sel;
    cfg->pool_size_mb = mem_menu.items[3].num_val;

    /* debug */
    cfg->debug_mode   = debug_menu.items[2].toggle;
    cfg->check_nan    = debug_menu.items[3].toggle;
    cfg->profile      = debug_menu.items[4].toggle;

    /* build */
    cfg->opt_level    = build_menu.items[2].radio_sel;
    cfg->enable_avx   = build_menu.items[3].toggle;
    cfg->enable_lto   = build_menu.items[4].toggle;
}

/* ── public API ──────────────────────────────────────────────────── */

void config_defaults(NeuralcConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->use_omp       = 1;
    cfg->omp_auto      = 1;
    cfg->omp_threads   = 4;
    cfg->use_gpu       = 0;
    cfg->use_blas      = 0;
    cfg->batch_size    = 64;
    cfg->learning_rate = 0.001f;
    cfg->optimizer     = 0;    /* Adam */
    cfg->dropout_rate  = 0.3f;
    cfg->epochs        = 20;
    cfg->grad_clip     = 1.0f;
    cfg->use_grad_clip = 1;
    cfg->allocator     = 0;    /* malloc */
    cfg->pool_size_mb  = 512;
    cfg->debug_mode    = 0;
    cfg->check_nan     = 0;
    cfg->profile       = 0;
    cfg->opt_level     = 2;    /* -O2 */
    cfg->enable_avx    = 0;
    cfg->enable_lto    = 0;
}

int config_ui_run(NeuralcConfig *cfg) {
    term_raw();
    term_size();
    build_menus(cfg);

    /* move cursor to first selectable item */
    root_menu.cursor = next_selectable(&root_menu, -1, +1);

    int saved = run_menu(&root_menu, "Main Menu");

    if (saved) {
        collect_config(cfg);
        printf(CLEAR_SCREEN);
        term_restore();
        return 0;
    }

    printf(CLEAR_SCREEN);
    term_restore();
    return 1;
}

int config_save(const NeuralcConfig *cfg, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "config_save: cannot open '%s'\n", path);
        return -1;
    }

    const char *opt_names[] = {"-O0", "-O1", "-O2", "-O3"};
    const char *opt_names2[] = {"ADAM", "SGD", "RMSPROP"};

    fprintf(f,
"/*\n"
" * neuralc_config.h — Auto-generated by neuralc config UI\n"
" * Do not edit manually — run: make config\n"
" */\n"
"#ifndef NEURALC_CONFIG_H\n"
"#define NEURALC_CONFIG_H\n\n"
"/* ── Performance ────────────────────────────────────── */\n"
"#define NEURALC_USE_OMP         %d\n"
"#define NEURALC_OMP_AUTO        %d\n"
"#define NEURALC_OMP_THREADS     %d\n"
"#define NEURALC_USE_GPU         %d\n"
"#define NEURALC_USE_BLAS        %d\n\n"
"/* ── Training Defaults ──────────────────────────────── */\n"
"#define NEURALC_BATCH_SIZE      %d\n"
"#define NEURALC_LR              %.6ff\n"
"#define NEURALC_OPTIMIZER       %s\n"
"#define NEURALC_DROPOUT         %.4ff\n"
"#define NEURALC_EPOCHS          %d\n"
"#define NEURALC_GRAD_CLIP       %.4ff\n"
"#define NEURALC_USE_GRAD_CLIP   %d\n\n"
"/* ── Memory ─────────────────────────────────────────── */\n"
"#define NEURALC_ALLOCATOR       %d\n"
"#define NEURALC_POOL_MB         %d\n\n"
"/* ── Debug ──────────────────────────────────────────── */\n"
"#define NEURALC_DEBUG           %d\n"
"#define NEURALC_CHECK_NAN       %d\n"
"#define NEURALC_PROFILE         %d\n\n"
"/* ── Build ──────────────────────────────────────────── */\n"
"#define NEURALC_OPT_LEVEL       \"%s\"\n"
"#define NEURALC_AVX             %d\n"
"#define NEURALC_LTO             %d\n\n"
"/* ── Derived flags for Makefile use ─────────────────── */\n"
"#if NEURALC_USE_OMP\n"
"  #ifndef USE_OMP\n"
"    #define USE_OMP\n"
"  #endif\n"
"#endif\n\n"
"#if NEURALC_USE_GPU\n"
"  #ifndef USE_OPENCL\n"
"    #define USE_OPENCL\n"
"  #endif\n"
"#endif\n\n"
"#endif /* NEURALC_CONFIG_H */\n",
        cfg->use_omp,
        cfg->omp_auto,
        cfg->omp_threads,
        cfg->use_gpu,
        cfg->use_blas,
        cfg->batch_size,
        cfg->learning_rate,
        opt_names2[cfg->optimizer < 3 ? cfg->optimizer : 0],
        cfg->dropout_rate,
        cfg->epochs,
        cfg->grad_clip,
        cfg->use_grad_clip,
        cfg->allocator,
        cfg->pool_size_mb,
        cfg->debug_mode,
        cfg->check_nan,
        cfg->profile,
        opt_names[cfg->opt_level < 4 ? cfg->opt_level : 2],
        cfg->enable_avx,
        cfg->enable_lto
    );

    fclose(f);
    return 0;
}

int config_load(NeuralcConfig *cfg, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;   /* file not found — use defaults */

    char line[256];
    config_defaults(cfg);

    while (fgets(line, sizeof(line), f)) {
        int  iv; float fv;
        if (sscanf(line, "#define NEURALC_USE_OMP %d",     &iv)==1) cfg->use_omp=iv;
        if (sscanf(line, "#define NEURALC_OMP_AUTO %d",    &iv)==1) cfg->omp_auto=iv;
        if (sscanf(line, "#define NEURALC_OMP_THREADS %d", &iv)==1) cfg->omp_threads=iv;
        if (sscanf(line, "#define NEURALC_USE_GPU %d",     &iv)==1) cfg->use_gpu=iv;
        if (sscanf(line, "#define NEURALC_USE_BLAS %d",    &iv)==1) cfg->use_blas=iv;
        if (sscanf(line, "#define NEURALC_BATCH_SIZE %d",  &iv)==1) cfg->batch_size=iv;
        if (sscanf(line, "#define NEURALC_LR %ff",         &fv)==1) cfg->learning_rate=fv;
        if (sscanf(line, "#define NEURALC_DROPOUT %ff",    &fv)==1) cfg->dropout_rate=fv;
        if (sscanf(line, "#define NEURALC_EPOCHS %d",      &iv)==1) cfg->epochs=iv;
        if (sscanf(line, "#define NEURALC_GRAD_CLIP %ff",  &fv)==1) cfg->grad_clip=fv;
        if (sscanf(line, "#define NEURALC_USE_GRAD_CLIP %d",&iv)==1) cfg->use_grad_clip=iv;
        if (sscanf(line, "#define NEURALC_ALLOCATOR %d",   &iv)==1) cfg->allocator=iv;
        if (sscanf(line, "#define NEURALC_POOL_MB %d",     &iv)==1) cfg->pool_size_mb=iv;
        if (sscanf(line, "#define NEURALC_DEBUG %d",       &iv)==1) cfg->debug_mode=iv;
        if (sscanf(line, "#define NEURALC_CHECK_NAN %d",   &iv)==1) cfg->check_nan=iv;
        if (sscanf(line, "#define NEURALC_PROFILE %d",     &iv)==1) cfg->profile=iv;
        if (sscanf(line, "#define NEURALC_AVX %d",         &iv)==1) cfg->enable_avx=iv;
        if (sscanf(line, "#define NEURALC_LTO %d",         &iv)==1) cfg->enable_lto=iv;
    }
    fclose(f);
    return 0;
}

void config_print(const NeuralcConfig *cfg) {
    const char *opts[] = {"Adam","SGD","RMSProp"};
    printf("\n=== neuralc Configuration ===\n");
    printf("Performance:\n");
    printf("  OpenMP:        %s", cfg->use_omp ? "enabled" : "disabled");
    if (cfg->use_omp)
        printf(" (%s, threads=%d)",
               cfg->omp_auto ? "auto" : "manual", cfg->omp_threads);
    printf("\n");
    printf("  GPU (OpenCL):  %s\n", cfg->use_gpu  ? "enabled" : "disabled");
    printf("  BLAS:          %s\n", cfg->use_blas ? "enabled" : "disabled");
    printf("Training:\n");
    printf("  Batch size:    %d\n", cfg->batch_size);
    printf("  Learning rate: %.6f\n", cfg->learning_rate);
    printf("  Optimizer:     %s\n",
           opts[cfg->optimizer < 3 ? cfg->optimizer : 0]);
    printf("  Dropout:       %.2f\n", cfg->dropout_rate);
    printf("  Epochs:        %d\n",   cfg->epochs);
    printf("  Grad clip:     %s (norm=%.2f)\n",
           cfg->use_grad_clip ? "yes" : "no", cfg->grad_clip);
    printf("Debug:\n");
    printf("  Debug mode:    %s\n", cfg->debug_mode ? "on" : "off");
    printf("  NaN check:     %s\n", cfg->check_nan  ? "on" : "off");
    printf("  Profiling:     %s\n", cfg->profile    ? "on" : "off");
    printf("Build:\n");
    const char *ol[] = {"-O0","-O1","-O2","-O3"};
    printf("  Opt level:     %s\n", ol[cfg->opt_level < 4 ? cfg->opt_level : 2]);
    printf("  AVX:           %s\n", cfg->enable_avx ? "yes" : "no");
    printf("  LTO:           %s\n", cfg->enable_lto ? "yes" : "no");
    printf("=============================\n\n");
}
