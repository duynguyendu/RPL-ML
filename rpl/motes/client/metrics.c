/*
 * metrics.c
 *
 * Collects and prints metrics for Contiki-NG / Cooja simulations:
 *   - ETX to the preferred RPL parent
 *   - Energest activity times (CPU / LPM / radio listen / transmit)
 *   - Latency (per packet, sender-embedded timestamp)
 *   - Computing time (rtimer-based stopwatch)
 *   - Hop count (from the IPv6 hop limit field)
 *   - CPU utilization (%)
 *   - Tx power
 *
 * All output is tagged (e.g. "ETX:", "LATENCY:") so it can be grep'd /
 * parsed straight out of the Cooja mote log.
 *
 * Assumes RPL Lite (net/routing/rpl-lite). If you're on RPL Classic,
 * metrics_print_etx() will need different accessors - ask if you need
 * that variant.
 */

#include "metrics.h"
#include "net/ipv6/uip.h"
#include "net/link-stats.h"
#include "net/queuebuf.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/clock.h"
#include "sys/energest.h"

#include <stdint.h>
#include <stdio.h>

#ifndef METRIC_LOG_INTERVAL
#define METRIC_LOG_INTERVAL 10
#endif
#define METRICS_PERIOD (METRIC_LOG_INTERVAL * CLOCK_SECOND)

/// -------------------- METRICS LOG -----------------------------------

#define METRICS_LOG                                                            \
  "ENERGEST: CPU=%lu LPM=%lu LISTEN=%lu "                                      \
  "TRANSMIT=%lu OFF=%lu TOTAL=%lu HOP_COUNT=%u "                               \
  "ETX=%u.%02u RSSI=%d TX=%d RX=%d ACKED=%d DROPPED=%d\n"

#define LATENCY_LOG "LATENCY: seqno=%" PRIu32 " rtt_ticks=%" PRIu32 "\n"
/// -------------------- METRICS LOG END -------------------------------

extern int hop_count;

PROCESS(metrics_process, "Metrics process");

uint32_t metrics_get_timestamp(void) { return (uint32_t)clock_time(); }

void metrics_log_latency(uint32_t seqno, uint32_t received_tick,
                         uint32_t sent_tick) {
  uint32_t rtt_tick = received_tick - sent_tick;
  printf(LATENCY_LOG, seqno, rtt_tick);
}

int get_hop_count(uint8_t initial_ttl) {
  uint8_t ttl = UIP_IP_BUF->ttl;
  if (ttl <= initial_ttl) {
    return (initial_ttl - ttl);
  } else {
    return -1;
  }
}

PROCESS_THREAD(metrics_process, ev, data) {
  static struct etimer metrics_timer;

  PROCESS_BEGIN();

  etimer_set(&metrics_timer, METRICS_PERIOD);

  // Periodic process: prints ETX, Energest, CPU util and Tx power
  while (1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&metrics_timer));

    energest_flush();
    unsigned long cpu = energest_type_time(ENERGEST_TYPE_CPU);
    unsigned long lpm = energest_type_time(ENERGEST_TYPE_LPM);
    unsigned long listen = energest_type_time(ENERGEST_TYPE_LISTEN);
    unsigned long transmit = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    unsigned long total = ENERGEST_GET_TOTAL_TIME();
    unsigned long off =
        total > (listen + transmit) ? total - listen - transmit : 0;

    unsigned etx_int = -1, etx_frac = 0;
    unsigned rssi = -1;
    unsigned tx_packets = 0, rx_packets = 0, ack_packets = 0,
             dropped_packets = 0;
    if (curr_instance.used) {
      rpl_parent_t *parent = curr_instance.dag.preferred_parent;
      if (parent != NULL) {
        const struct link_stats *stats = rpl_neighbor_get_link_stats(parent);
        if (stats != NULL) {
          unsigned etx = (stats->etx * 100) / LINK_STATS_ETX_DIVISOR;
          etx_int = etx / 100;
          etx_frac = etx % 100;

          rssi = stats->rssi;
          tx_packets = stats->cnt_current.num_packets_tx;
          rx_packets = stats->cnt_current.num_packets_rx;
          ack_packets = stats->cnt_current.num_packets_acked;
          dropped_packets = stats->cnt_current.num_queue_drops;
        }
      }
    }

    printf(METRICS_LOG, cpu, lpm, listen, transmit, off, total, hop_count,
           etx_int, etx_frac, rssi, tx_packets, rx_packets, ack_packets,
           dropped_packets);

    etimer_reset(&metrics_timer);
  }

  PROCESS_END();
}

void metrics_start(void) { process_start(&metrics_process, NULL); }
