#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <papi.h>

#define REPS 1024*1024

__global__ void chase(unsigned long* dst, unsigned long *src, unsigned long N, unsigned int stride, int print_chain);

extern "C" void gpu_mem(int EventSet, FILE *ofp_papi, unsigned int N, int tpb) {

    int retval, print_chain = 0;
    long long values = 0;
    unsigned long *avail, *h_array, *d_res, *d_inp;
    unsigned long i, j, buffer_size, stride, remainingElemCnt, currIndex, elemCnt, index, uniqIndex;
    hipError_t status;

    buffer_size = N;
    stride = tpb;

    dim3 threads_per_block( tpb, 1, 1 );
    dim3 blocks_in_grid( 1, 1, 1 );

    if( buffer_size/tpb < 2 ) {
        fprintf(stderr, "buffer_size / tpb must be over 2.\n");
    }

    /* Allocate device arrays:
     * Allocate a few more elements to send meta-data out of the GPU.
     */
    status = hipMalloc((void**)&d_res, (buffer_size+32)*sizeof(unsigned long));
    if( status != hipSuccess ) {
        fprintf(stderr, "Could not allocate device output array.\n");
        return;
    }

    status = hipMalloc((void**)&d_inp, buffer_size*sizeof(unsigned long));
    if( status != hipSuccess ) {
        fprintf(stderr, "Could not allocate device input array.\n");
        return;
    }

    /* Allocate host arrays:
     * Allocate a few more elements to send meta-data out of the GPU.
     */
    status = hipHostMalloc((void**)&h_array, (buffer_size+32)*sizeof(unsigned long), 0);
    if(status != hipSuccess) {
        fprintf(stderr, "Could not allocate host memory.\n");
        return;
    }

    avail = (unsigned long *)malloc(buffer_size/stride*sizeof(unsigned long));
    if( NULL == avail ) {
        fprintf(stderr, "Could not allocate host memory.\n");
        return;
    }

    /* Generate the pointer chain. */
    for( i=0; i<buffer_size/stride; i++ ) {
        avail[i] = i;
    }

    remainingElemCnt = buffer_size/stride;
    currIndex=0;

    srandom(1);
    for( elemCnt=0; elemCnt<buffer_size/stride-1; ++elemCnt ) {
        // We add 1 (and subtract 1 from the modulo divisor) because the first
        //element (0) is the currIndex in the first iteration, so it can't be in
        // the list of available elements.
        index = 1+random() % (remainingElemCnt-1);
        uniqIndex = stride*avail[index];
        // replace the chosen number with the last element.
        avail[index] = avail[remainingElemCnt-1];
        // shrink the effective array size so the last element "drops off".
        remainingElemCnt--;

        for( j=0; j<tpb; j++ ) {
            h_array[currIndex+j] = uniqIndex+j;
        }

        currIndex = uniqIndex;
    }

    // close the circle by making the last element(s) point to the zero-th element(s)
    for(unsigned long j=0; j<tpb; j++)
        h_array[currIndex+j] = 0+j;

    /* Copy host data to device:
     * We have allocate a few more elements to send meta-data out of the GPU,
     * but they don't contain any values, so we don't need to copy them.
     */
    status = hipMemcpy(d_inp, h_array, buffer_size*sizeof(unsigned long), hipMemcpyHostToDevice);
    if(status != hipSuccess){
        fprintf(stderr, "Could not copy memory to device.\n");
        return;
    }

    /* Start PAPI counters. */
    if( (retval = PAPI_start( EventSet )) != PAPI_OK ) return;

    /* Various floating-point operation kernels. */
    //rocblas_handle handle;
    //rocblas_status rstatus;
    //double alpha = 1.0, beta = 1.0;
    //rstatus = rocblas_create_handle(&handle);
    //rstatus = RBLAS_GEMM(handle, rocblas_operation_none, rocblas_operation_none, X, X, Y, &alpha, devA, X, devB, Y, &beta, devC, X);
    //rstatus = rocblas_destroy_handle(handle);
    hipLaunchKernelGGL(chase, blocks_in_grid, threads_per_block, 0, 0, d_res, d_inp, buffer_size, stride, print_chain);

    /* Error checking. */
    status = hipGetLastError();
    if( hipSuccess != status ) {
        fprintf(stderr, "Error 1.\n");
        return;
    }
    status = hipDeviceSynchronize();
    if( hipSuccess != status ) {
        fprintf(stderr, "Error 2.\n");
        return;
    }

    /* Stop PAPI counters. */
    if( (retval = PAPI_stop(EventSet, &values)) != PAPI_OK ) return;
    fprintf(ofp_papi, "%lu %lf\n", buffer_size*sizeof(unsigned long), ((double)values)/((double)(REPS*tpb)));

    /* Copy device data to host. */
    status = hipMemcpy(h_array, d_res, (buffer_size+32)*sizeof(unsigned long), hipMemcpyDeviceToHost);
    if(status != hipSuccess){
        fprintf(stderr, "Could not memory to host.\n");
        return;
    }

    /* Free device memory. */
    status = hipFree(d_inp);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device input array.\n");
        return;
    }

    status = hipFree(d_res);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device output array.\n");
        return;
    }

    /* Free host memory. */
    status = hipFree(h_array);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host memory.\n");
        return;
    }

    free(avail);

    return;
}

__global__ void chase(unsigned long* dst, unsigned long *src, unsigned long N, unsigned int stride, int print_chain) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tpb = blockDim.x;
    unsigned long idx, start_time, end_time, max_repeat;

    idx = src[tid];

    if( print_chain ){
        dst[idx] = src[idx];
        for(int j=0; j<(N/tpb); j++){
            idx = src[idx];
            dst[idx] = src[idx];
        }
        return;
    }

    max_repeat = REPS;

    __asm__ volatile ("s_memrealtime %0\n s_waitcnt lgkmcnt(0)" : "=s" (start_time) );

    for(int j=0; j<max_repeat; j+=16){
        // Have a side-effect so the compiler does not throw our code away.
        if( !(j%(31*16)) )
            dst[idx] = src[idx];

        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];

        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
        idx = src[idx];
    }

    __asm__ volatile ("s_memrealtime %0\n s_waitcnt lgkmcnt(0)" : "=s" (end_time) );

    dst[0] = max_repeat*tpb;
    dst[1+tid] = start_time;
    dst[1+tpb+tid] = end_time;
}
