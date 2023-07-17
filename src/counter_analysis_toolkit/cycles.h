#ifndef _CYCLES_
#define _CYCLES_

#include <stdio.h>
#include "hw_desc.h"

void monitor_core_freqs();

void monitor_cycles_idle();
void monitor_cycles_nominal();
void monitor_cycles_busy();

void monitor_timing_idle();
void monitor_timing_nominal();
void monitor_timing_busy();

void monitor_events_idle();
void monitor_events_nominal();
void monitor_events_busy();

int  open_freq_files( int *freq_fds );
void read_freq_files( char *buffer, int *freq_fds, double *freqSums );
void close_freq_files( char *buffer, int *freq_fds );

void get_cycles( int kernel );
void cycles_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir);

#endif
