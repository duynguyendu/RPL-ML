/*
 * metrics.h
 *
 * Declarations for the metrics collection module (ETX, energy, latency,
 * computing time, hop count, CPU utilization, Tx power).
 *
 * Include this header in any .c file that needs to log a metric, and add
 * metrics.c to your Makefile's CONTIKI_SOURCEFILES (or PROJECT_SOURCEFILES).
 */

#pragma once

#include <stdint.h>

void metrics_start(void);

void metrics_print_dodag(void);

int get_hop_count(uint8_t initial_ttl);

// Per-packet metrics
uint32_t metrics_get_timestamp(void);

void metrics_log_latency(uint32_t seqno, uint32_t received_tick, uint32_t sent_tick);
