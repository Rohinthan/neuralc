#ifndef NEURALC_INIT_H
#define NEURALC_INIT_H

/*
 * neuralc_init.h — Runtime initialization
 *
 * Call neuralc_init() at the start of main() to apply
 * settings from neuralc_config.h (thread count, debug mode etc.)
 *
 * On GCC/Clang this is called automatically — no user code needed.
 */
void neuralc_init(void);

#endif /* NEURALC_INIT_H */
