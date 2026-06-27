/*
 * neuralc_init.c — Auto-apply neuralc_config.h settings at startup
 *
 * This file reads neuralc_config.h and applies settings
 * automatically when any neuralc program starts.
 *
 * Included in all neuralc builds via Makefile.
 */

#include <stdio.h>
#include <stdlib.h>

/* Load generated config if available */
#ifdef NEURALC_HAS_CONFIG
#include "neuralc_config.h"
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

/*
 * neuralc_init — call this at the start of main()
 * OR it is called automatically via the constructor below.
 *
 * Sets:
 *   - OpenMP thread count (from config or environment)
 *   - Any other runtime settings from neuralc_config.h
 */
void neuralc_init(void) {
#ifdef USE_OMP

#ifdef NEURALC_HAS_CONFIG
    /* Use thread count from config UI */
    if (NEURALC_USE_OMP) {
        if (NEURALC_OMP_AUTO) {
            /* Auto: let OpenMP use all available cores */
            int cores = omp_get_max_threads();
            omp_set_num_threads(cores);
        } else {
            /* Manual: use exact count from config */
            omp_set_num_threads(NEURALC_OMP_THREADS);
        }
    }
#else
    /* No config file — use OMP_NUM_THREADS env or all cores */
    const char *env = getenv("OMP_NUM_THREADS");
    if (env) {
        int t = atoi(env);
        if (t > 0) omp_set_num_threads(t);
    }
#endif /* NEURALC_HAS_CONFIG */

#ifdef NEURALC_HAS_CONFIG
    if (NEURALC_DEBUG) {
        int threads = omp_get_max_threads();
        printf("[neuralc] OpenMP: %d thread(s) — %s mode\n",
               threads,
               NEURALC_OMP_AUTO ? "auto" : "manual");
    }
#endif

#endif /* USE_OMP */

#ifdef NEURALC_HAS_CONFIG
    #ifdef NEURALC_DEBUG
    if (NEURALC_DEBUG)
        printf("[neuralc] Debug mode ON\n");
    #endif
#endif
}

/*
 * __attribute__((constructor)) — runs neuralc_init() automatically
 * before main() on GCC/Clang. No user code change needed.
 */
#ifdef __GNUC__
__attribute__((constructor))
static void _neuralc_auto_init(void) {
    neuralc_init();
}
#endif
