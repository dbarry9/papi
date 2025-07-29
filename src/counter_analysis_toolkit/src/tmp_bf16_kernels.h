#if defined(X86)
#if defined(AVX128_AVAIL)
void tmp_test_bf16_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
#if defined(AVX256_AVAIL)
void tmp_test_bf16_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
#if defined(AVX512_AVAIL)
void tmp_test_bf16_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
#endif
#endif
#endif
#endif
