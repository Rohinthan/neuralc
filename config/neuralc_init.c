-/*
 * src/neuralc_init.c — Auto-apply neuralc_config.h settings at startup
 *
 * This file reads neuralc_config.h and applies settings
 * automatically when any neuralc program starts.
 */

#include <stdio.h>
#include <stdlib.h>

/*
 * Only attempt inclusion if NEURALC_HAS_CONFIG flag was injected by the Makefile */
#ifdef NEURALC_HAS_CONFIG
  #include "neuralc_config.h"
#endif

#ifdef USE_OMP
  #include <omp.h>
#endif

/*
 * neuralc_init — applies setup attributes at runtime
 */
void neuralc_init(void) {
#ifdef USE_OMP

#ifdef NEURALC_HAS_CONFIG
    /* Safely fall back if variables are missing from an old header format */
    #ifdef NEURALC_USE_OMP
    if (NEURALC_USE_OMP) {
        #ifdef NEURALC_OMP_AUTO
        if (NEURALC_OMP_AUTO) {
            int cores = omp_get_max_threads();
            omp_set_num_threads(cores);
        } else {
            #ifdef NEURALC_OMP_THREADS
            omp_set_num_threads(NEURALC_OMP_THREADS);
            #endif
        }
        #endif
    }
    #endif
#else
    /* Safe fallback if no config file has been generated yet */
    const char *env = getenv("OMP_NUM_THREADS");
    if (env) {
        int t = atoi(env);
        if (t > 0) omp_set_num_threads(t);
    }
#endif /* NEURALC_HAS_CONFIG */

#ifdef NEURALC_HAS_CONFIG
    #if defined(NEURALC_DEBUG) && defined(NEURALC_OMP_AUTO)
    if (NEURALC_DEBUG) {
        int threads = omp_get_max_threads();
        printf("[neuralc] OpenMP: %d thread(s) — %s mode\n",
               threads,
               NEURALC_OMP_AUTO ? "auto" : "manual");
    }
    #endif
#endif

#endif /* USE_OMP */

#ifdef NEURALC_HAS_CONFIG
    #ifdef NEURALC_DEBUG
    if (NEURALC_DEBUG) {
        printf("[neuralc] Debug mode ON\n");
    }
    #endif
#endif
}

/*
 * Auto-execute initialization prior to main entry point
 */
#ifdef __GNUC__
__attribute__((constructor))
static void _neuralc_auto_init(void) {
    neuralc_init();
}
#endif
