/*
 * metrics.h
 *
 * Declarations for the metrics collection module (ETX, energy, latency,
 * computing time, hop count, CPU utilization, Tx power).
 *
 * Include this header in any .c file that needs to log a metric, and add
 * metrics.c to your Makefile's CONTIKI_SOURCEFILES (or PROJECT_SOURCEFILES).
 */

#ifndef METRICS_H_
#define METRICS_H_

#include <stdint.h>
#include "sys/rtimer.h"

/* Period (in seconds) for the periodic metrics process (ETX, energy, CPU,
 * Tx power). Override in project-conf.h if needed. */
#ifndef METRICS_PERIOD
#define METRICS_PERIOD 30
#endif

void metrics_start(void);

/* --- One-shot / periodic metrics -------------------------------------- */

/* Prints ETX to the current preferred RPL parent. */
void metrics_print_etx(void);

/* Prints DODAG info: instance id, DAG ID, version, rank, grounded flag,
 * role (root/node), and preferred parent address (if any). */
void metrics_print_dodag(void);

/* Prints Energest breakdown (CPU/LPM/Deep LPM/TX/RX ticks), the tick rate and CPU util %. */
void metrics_energest(void);

/* Prints the radio's current Tx power (dBm or raw driver units). */
void metrics_print_txpower(void);

/* --- Per-packet metrics ------------------------------------------------ */

/* Call at the SENDER right before sending, to get a timestamp to embed in
 * the packet payload. */
uint32_t metrics_get_timestamp(void);

/* Call at the RECEIVER with the sequence number and the timestamp that was
 * embedded in the packet (as returned by metrics_get_timestamp() at the
 * sender). Prints the one-way latency in clock ticks and milliseconds. */
void metrics_log_latency(uint16_t seqno, uint32_t sent_timestamp);

/* Call at the RECEIVER inside the UDP/packet callback, while uip_buf still
 * holds the just-received IPv6 packet, to print the hop count travelled.
 * initial_ttl should be the hop limit the sender used (usually UIP_TTL,
 * i.e. 64, unless you changed it). */
void metrics_print_hop_count(uint8_t initial_ttl);

/* --- Computing time (wrap any block of code you want to time) --------- */

/* Starts a computing-time measurement. Not reentrant/nestable - use one
 * timer at a time, or keep your own rtimer_clock_t if you need nesting. */
void metrics_time_start(void);

/* Ends a computing-time measurement started with metrics_time_start() and
 * prints the elapsed time in rtimer ticks and microseconds, tagged with
 * "label". */
void metrics_time_end(const char *label);

#endif /* METRICS_H_ */
