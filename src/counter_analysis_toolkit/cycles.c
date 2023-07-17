#define _GNU_SOURCE
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <omp.h>
#include <fcntl.h>
#include <papi.h>

#include "caches.h"
#include "cycles.h"

//#define NUM_THRDS 4
#define PERIOD 1000000
#define LCLBUFSIZ 64
#define GEMMSIZE 1024
#define GEMMITER 5
#define GEMVSIZE 720
#define GEMVITER 50
#define OFFSET 3

#define IDLE    0
#define NOMINAL 1
#define BUSY    2

useconds_t sample = PERIOD;

int EventSet;
int numCores;
long long values[1];
long long refCyc;
double refTime;
double *freqSums;
char *gbl_event_name;
FILE *gbl_ofp_papi;
volatile int g_done = 0;

static void print_header( char *kernel, FILE *fp );
static void resultline( FILE *fp );

int open_freq_files( int *freq_fds ) {

    int closingCtr;
    int fileCtr;
    char fileName[LCLBUFSIZ];

    for( fileCtr = 0; fileCtr < numCores; ++fileCtr ) {

        sprintf(fileName, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", fileCtr);
        freq_fds[fileCtr] = open(fileName, O_SYNC|O_RDONLY);

        if( 0 > freq_fds[fileCtr] ) {
            fprintf(stderr, "Could not open %s\n", fileName);
            for( closingCtr = 0; closingCtr < fileCtr; ++closingCtr ) {
                close(freq_fds[fileCtr]);
            }
            return -1;
        }
    }

    return 0;
}

void read_freq_files( char *buffer, int *freq_fds, double *freqSums ) {

    int fileCtr, sz;

    for( fileCtr = 0; fileCtr < numCores; ++fileCtr ) {
        sz = pread(freq_fds[fileCtr], buffer, LCLBUFSIZ, 0);
        buffer[sz] = '\0';

        freqSums[fileCtr] += (double)atoi(buffer);
    }

    return;
}

void close_freq_files( char *buffer, int *freq_fds ) {

    int fileCtr;

    for( fileCtr = 0; fileCtr < numCores; ++fileCtr ) {
        if( 0 > close(freq_fds[fileCtr]) ) {
            fprintf(stderr, "Could not close frequency file for core %d\n", fileCtr);
        }
    }

    free(buffer);
    free(freq_fds);

    return;
}

void monitor_core_freqs() {

    int coreIdx, iterCtr = 0;

    /* Allocate space for the contents, number, and average values of the system files. */
    char *buffer = (char*)malloc(LCLBUFSIZ*sizeof(char));
    if( NULL == buffer ) {
        fprintf(stderr, "Cannot allocate space for reading content from sys files.\n");
        return;
    }

    int *freq_fds = (int*)malloc(numCores*sizeof(int*));
    if( NULL == freq_fds ) {
        fprintf(stderr, "Could not allocate space for the core frequency files.\n");
        return;
    }

    /* Open the file descriptors. */
    open_freq_files( freq_fds );

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    /* Periodically read system file contents. */
    while( !g_done ) {

        read_freq_files( buffer, freq_fds, freqSums );

        usleep(1);
        ++iterCtr;
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier

    /* Close the file descriptors. */
    close_freq_files( buffer, freq_fds );

    /* Compute the average frequency value.  */
    for( coreIdx = 0; coreIdx < numCores; ++coreIdx) {
        freqSums[coreIdx] /= ((double)iterCtr);
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_cycles_idle() {

    long long tstart, tstop;

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = PAPI_get_real_cyc();

    /* Performance-critical workload. */
    usleep(5*PERIOD);

    tstop = PAPI_get_real_cyc();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refCyc = tstop-tstart;

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_cycles_nominal() {

    long long tstart, tstop;
    double *A, *B;
    int iter = 0, i, j, k;
    double sum = 0.0;

    /* Allocate and initialize data for GEMV. */
    A = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));
    B = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));

    for( i = 0; i < GEMVSIZE; ++i ) {
        for( j = 0; j < GEMVSIZE; ++j ) {
            for( k = 0; k < GEMVSIZE; ++k ) {
                A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
                B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            }
        }
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = PAPI_get_real_cyc();

    /* Performance-critical workload. */
    /*while( iter < PERIOD/10000 ) {
        sum += ((double)random())/((double)RAND_MAX) * (double)1.1;
        //usleep(1);
        ++iter;
    }*/
    for( iter = 0; iter < GEMVITER; ++iter ) {
        for( i = 0; i < GEMVSIZE; ++i ) {
            for( j = 0; j < GEMVSIZE; ++j ) {
                for( k = 0; k < GEMVSIZE; ++k ) {
                    //B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    B[k*GEMVSIZE*GEMVSIZE+i*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[i*GEMVSIZE*GEMVSIZE+k*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] = A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)];
                    /*B[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] = \
                      (A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET-1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET+1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)])/3.0;
		      */
                }
            }
        }
    }

    tstop = PAPI_get_real_cyc();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refCyc = tstop-tstart;

    /* Use the computed GEMV result so the compiler won't throw it away. */
    i = GEMVSIZE/2 * GEMVSIZE + GEMVSIZE - 1;
    if ( ((int)(A[i]+B[i]+sum))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_cycles_busy() {

    long long tstart, tstop;
    double *A, *B, *C;
    int iter, i, j, k;
    double sum;

    /* Allocate and initialize data for GEMM. */
    A = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    B = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    C = (double*)calloc(GEMMSIZE*GEMMSIZE, sizeof(double));

    for( i = 0; i < GEMMSIZE; ++i ) {
        for( j = 0; j < GEMMSIZE; ++j ) {
            A[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            B[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
        }
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = PAPI_get_real_cyc();

    /* Performance-critical workload. */
    for( iter = 0; iter < GEMMITER; ++iter ) {
        for( i = 0; i < GEMMSIZE; ++i ) {
            for( j = 0; j < GEMMSIZE; ++j ) {
                sum = 0.0;
                for( k = 0; k < GEMMSIZE; ++k ) {
                    sum += A[i*GEMMSIZE+k] * B[k*GEMMSIZE+j];
                }
                C[i*GEMMSIZE+j] = sum;
            }
        }
    }

    tstop = PAPI_get_real_cyc();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refCyc = tstop-tstart;

    /* Use the computed GEMM result so the compiler won't throw it away. */
    i = GEMMSIZE/2 * GEMMSIZE + GEMMSIZE - 1;
    if ( ((int)(A[i]+B[i]+C[i]))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);
    free(C);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_timing_idle() {

    double tstart, tstop;

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = getticks();

    /* Performance-critical workload. */
    usleep(5*PERIOD);

    tstop = getticks();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refTime = elapsed(tstop, tstart)*1.0e-6;

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_timing_nominal() {

    double tstart, tstop;
    double *A, *B;
    int iter = 0, i, j, k;
    double sum = 0.0;

    /* Allocate and initialize data for GEMV. */
    A = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));
    B = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));

    for( i = 0; i < GEMVSIZE; ++i ) {
        for( j = 0; j < GEMVSIZE; ++j ) {
            for( k = 0; k < GEMVSIZE; ++k ) {
                A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
                B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            }
        }
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = getticks();

    /* Performance-critical workload. */
    /*while( iter < PERIOD/10000 ) {
        sum += ((double)random())/((double)RAND_MAX) * (double)1.1;
        //usleep(1);
        ++iter;
    }*/
    for( iter = 0; iter < GEMVITER; ++iter ) {
        for( i = 0; i < GEMVSIZE; ++i ) {
            for( j = 0; j < GEMVSIZE; ++j ) {
                for( k = 0; k < GEMVSIZE; ++k ) {
                    //B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    B[k*GEMVSIZE*GEMVSIZE+i*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[i*GEMVSIZE*GEMVSIZE+k*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    /*B[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] = \
                      (A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET-1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET+1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)])/3.0;
		      */
                }
            }
        }
    }

    tstop = getticks();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refTime = elapsed(tstop, tstart)*1.0e-6;

    /* Use the computed GEMV result so the compiler won't throw it away. */
    i = GEMVSIZE/2 * GEMVSIZE + GEMVSIZE - 1;
    if ( ((int)(A[i]+B[i]+sum))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_timing_busy() {

    double tstart, tstop;
    double *A, *B, *C;
    int iter, i, j, k;
    double sum;

    /* Allocate and initialize data for GEMM. */
    A = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    B = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    C = (double*)calloc(GEMMSIZE*GEMMSIZE, sizeof(double));

    for( i = 0; i < GEMMSIZE; ++i ) {
        for( j = 0; j < GEMMSIZE; ++j ) {
            A[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            B[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
        }
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    tstart = getticks();

    /* Performance-critical workload. */
    for( iter = 0; iter < GEMMITER; ++iter ) {
        for( i = 0; i < GEMMSIZE; ++i ) {
            for( j = 0; j < GEMMSIZE; ++j ) {
                sum = 0.0;
                for( k = 0; k < GEMMSIZE; ++k ) {
                    sum += A[i*GEMMSIZE+k] * B[k*GEMMSIZE+j];
                }
                C[i*GEMMSIZE+j] = sum;
            }
        }
    }

    tstop = getticks();

    /* Wait for each thread to get here. */
    #pragma omp barrier

    refTime = elapsed(tstop, tstart)*1.0e-6;

    /* Use the computed GEMM result so the compiler won't throw it away. */
    i = GEMMSIZE/2 * GEMMSIZE + GEMMSIZE - 1;
    if ( ((int)(A[i]+B[i]+C[i]))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);
    free(C);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_events_idle() {

    int retval = PAPI_OK;

    /* Print kernel header. */
    print_header("Idle Test", gbl_ofp_papi);

    /* Reset the flag to periodically monitor system files. */
    g_done = 0;

    /* Set-up the event set. */
    EventSet = PAPI_NULL;
    if ( (retval = PAPI_create_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not create event set.\n");
        return;
    }

    if ( (retval = PAPI_add_named_event( EventSet, gbl_event_name )) != PAPI_OK ) {
        fprintf(stderr, "Could add event to event set.\n");
        return;
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    /* Start PAPI counters. */
    if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "PAPI_start() failed.\n");
        return;
    }

    /* Performance-critical workload. */
    usleep(5*PERIOD);

    /* Signal to end monitoring of CPU frequency files. */
    g_done = 1;

    /* Wait for each thread to get here. */
    #pragma omp barrier

    /* Stop PAPI counters. */
    if ( (retval = PAPI_stop(EventSet, &(values[0]))) != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop() failed.\n");
        return;
    }

    /* Clean-up. */
    if ( (retval = PAPI_cleanup_eventset( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not clean-up event set.\n");
        return;
    }

    if ( (retval = PAPI_destroy_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not destroy event set.\n");
        return;
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_events_nominal() {

    int retval = PAPI_OK;
    double *A, *B;
    int iter = 0, i, j, k;
    double sum = 0.0;

    /* Allocate and initialize data for GEMV. */
    A = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));
    B = (double*)malloc(GEMVSIZE*GEMVSIZE*GEMVSIZE*sizeof(double));

    for( i = 0; i < GEMVSIZE; ++i ) {
        for( j = 0; j < GEMVSIZE; ++j ) {
            for( k = 0; k < GEMVSIZE; ++k ) {
                A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
                B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            }
        }
    }

    /* Print kernel header. */
    print_header("Nominal Test", gbl_ofp_papi);

    /* Reset the flag to periodically monitor system files. */
    g_done = 0;

    /* Set-up the event set. */
    EventSet = PAPI_NULL;
    if ( (retval = PAPI_create_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not create event set.\n");
        return;
    }

    if ( (retval = PAPI_add_named_event( EventSet, gbl_event_name )) != PAPI_OK ) {
        fprintf(stderr, "Could not add event to event set.\n");
        return;
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    /* Start PAPI counters. */
    if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "PAPI_start() failed.\n");
        return;
    }

    /* Performance-critical workload. */
    /*while( iter < PERIOD/10000 ) {
        sum += ((double)random())/((double)RAND_MAX) * (double)1.1;
        //usleep(1);
        ++iter;
    }*/
    for( iter = 0; iter < GEMVITER; ++iter ) {
        for( i = 0; i < GEMVSIZE; ++i ) {
            for( j = 0; j < GEMVSIZE; ++j ) {
                for( k = 0; k < GEMVSIZE; ++k ) {
                    //B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    B[k*GEMVSIZE*GEMVSIZE+i*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[i*GEMVSIZE*GEMVSIZE+k*GEMVSIZE+j] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k] = A[i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k];
                    //B[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] = A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)];
                    /*B[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] = \
                      (A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET-1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)] + \
                      A[(i*GEMVSIZE*GEMVSIZE+j*GEMVSIZE+k+OFFSET+1)%(GEMVSIZE*GEMVSIZE*GEMVSIZE)])/3.0;
		      */
                }
            }
        }
    }

    /* Signal to end monitoring of CPU frequency files. */
    g_done = 1;

    /* Wait for each thread to get here. */
    #pragma omp barrier

    /* Stop PAPI counters. */
    if ( (retval = PAPI_stop(EventSet, &(values[0]))) != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop() failed.\n");
        return;
    }

    /* Clean-up. */
    if ( (retval = PAPI_cleanup_eventset( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not clean-up event set.\n");
        return;
    }

    if ( (retval = PAPI_destroy_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not destroy event set.\n");
        return;
    }

    /* Use the computed GEMV result so the compiler won't throw it away. */
    i = GEMVSIZE/2 * GEMVSIZE + GEMVSIZE - 1;
    if ( ((int)(A[i]+B[i]+sum))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void monitor_events_busy() {

    int retval = PAPI_OK;
    double *A, *B, *C;
    int iter, i, j, k;
    double sum;

    /* Print kernel header. */
    print_header("Busy Test", gbl_ofp_papi);

    /* Reset the flag to periodically monitor system files. */
    g_done = 0;

    /* Allocate and initialize data for GEMM. */
    A = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    B = (double*)malloc(GEMMSIZE*GEMMSIZE*sizeof(double));
    C = (double*)calloc(GEMMSIZE*GEMMSIZE, sizeof(double));

    for( i = 0; i < GEMMSIZE; ++i ) {
        for( j = 0; j < GEMMSIZE; ++j ) {
            A[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            B[i*GEMMSIZE+j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
        }
    }

    /* Set-up the event set. */
    EventSet = PAPI_NULL;
    if ( (retval = PAPI_create_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not create event set.\n");
        return;
    }

    if ( (retval = PAPI_add_named_event( EventSet, gbl_event_name )) != PAPI_OK ) {
        fprintf(stderr, "Could not add event to event set.\n");
        return;
    }

    /* Wait for each thread to get here. */
    #pragma omp barrier
    usleep(10*PERIOD);
    #pragma omp barrier

    /* Start PAPI counters. */
    if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "PAPI_start() failed.\n");
        return;
    }

    /* Performance-critical workload. */
    for( iter = 0; iter < GEMMITER; ++iter ) {
        for( i = 0; i < GEMMSIZE; ++i ) {
            for( j = 0; j < GEMMSIZE; ++j ) {
                sum = 0.0;
                for( k = 0; k < GEMMSIZE; ++k ) {
                    sum += A[i*GEMMSIZE+k] * B[k*GEMMSIZE+j];
                }
                C[i*GEMMSIZE+j] = sum;
            }
        }
    }

    /* Signal to end monitoring of CPU frequency files. */
    g_done = 1;

    /* Wait for each thread to get here. */
    #pragma omp barrier

    /* Stop PAPI counters. */
    if ( (retval = PAPI_stop(EventSet, &(values[0]))) != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop() failed.\n");
        return;
    }

    /* Clean-up. */
    if ( (retval = PAPI_cleanup_eventset( EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not cleanup event set.\n");
        return;
    }

    if ( (retval = PAPI_destroy_eventset( &EventSet )) != PAPI_OK ) {
        fprintf(stderr, "Could not destroy event set.\n");
        return;
    }

    /* Use the computed GEMM result so the compiler won't throw it away. */
    i = GEMMSIZE/2 * GEMMSIZE + GEMMSIZE - 1;
    if ( ((int)(A[i]+B[i]+C[i]))%5 == 0 ) {
        fprintf(stderr, "Random side-effect from benchmark. Please disregard.\n");
    }

    /* Free dynamically allocated memory. */
    free(A);
    free(B);
    free(C);

    /* Wait for each thread to get here. */
    #pragma omp barrier

    return;
}

void get_cycles( int kernel ) {

    /* Allocate memory for monitoring frequency. */
    freqSums = (double*)calloc(numCores, sizeof(double));
    if( NULL == freqSums ) {
        fprintf(stderr, "Could not allocate space for the core frequency data.\n");
        return;
    }

    /* Spawn the three threads. */
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            switch(kernel) {
              case IDLE:
                  monitor_timing_idle(NULL);
                  break;
              case NOMINAL:
                  monitor_timing_nominal(NULL);
                  break;
              case BUSY:
                  monitor_timing_busy(NULL);
                  break;
              default:
                  break;
            }
        }
        #pragma omp section
        {
            switch(kernel) {
              case IDLE:
                  monitor_cycles_idle(NULL);
                  break;
              case NOMINAL:
                  monitor_cycles_nominal(NULL);
                  break;
              case BUSY:
                  monitor_cycles_busy(NULL);
                  break;
              default:
                  break;
            }
        }
        #pragma omp section
        {
            switch(kernel) {
              case IDLE:
                  monitor_events_idle(NULL);
                  break;
              case NOMINAL:
                  monitor_events_nominal(NULL);
                  break;
              case BUSY:
                  monitor_events_busy(NULL);
                  break;
              default:
                  break;
            }
        }
        #pragma omp section
        {
            monitor_core_freqs(NULL);
        }
    }

    /* Print the result to the file. */
    resultline(gbl_ofp_papi);

    /* Free dynamically allocated memory. */
    free(freqSums);

    return;
}

static void print_header( char* kernel, FILE *fp ) {

    int coreIdx;

    fprintf(fp, "#%s\n", kernel);
    fprintf(fp, "#RawEvtCnt NormEvtCnt(GEvt/s) RawRefCyc NormRefCyc(GHz)");
    for( coreIdx = 0; coreIdx < numCores; ++coreIdx) {
        fprintf(fp, " Avg%d", coreIdx);
    }
    fprintf(fp, " (GHz)\n");
}

static void resultline( FILE *fp ) {

    int coreIdx;

    fprintf(fp, "%lld %.4lf %lld %.4lf ", values[0], ((double)values[0])/refTime*1.0e-9, refCyc, refCyc/refTime*1.0e-9);
    for( coreIdx = 0; coreIdx < numCores; ++coreIdx) {
        fprintf(fp, " %.4lf", freqSums[coreIdx]*1.0e-6);
    }
    fprintf(fp, "\n");
}

void cycles_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    const char *sufx = ".cycles";
    char *papiFileName = NULL;
    FILE *ofp_papi;

    /* Get the number of physical cores. */
    numCores = hw_desc->numcpus;
    if( numCores <= 0 ) {
        numCores = 1;
    }
    /* Create output file name. */
    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char) ))) {
        fprintf(stderr, "Failed to allocate papiFileName.\n");
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        fprintf(stderr, "sprintf failed to copy into papiFileName.\n");
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }

    /* Track event name and file globally so that threads can read them. */
    gbl_event_name = papi_event_name;
    gbl_ofp_papi = ofp_papi;

    /* Run cycles benchmark kernels. */
    get_cycles( IDLE );
    //get_cycles( NOMINAL );
    get_cycles( BUSY );

    /* Close output file. */
    fclose(ofp_papi);
error0:
    free(papiFileName);

    return;
}
