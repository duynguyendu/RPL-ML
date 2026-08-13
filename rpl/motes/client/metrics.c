/*
 * metrics.c
 *
 * Collects and prints metrics for Contiki-NG / Cooja simulations:
 *   - ETX to the preferred RPL parent
 *   - Energy consumption (via Energest)
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
#include "net/routing/rpl-lite/rpl-dag-root.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/clock.h"
#include "sys/energest.h"
#include "utils.h"

#include <stdint.h>
#include <stdio.h>

#define METRICS_PERIOD 10 * CLOCK_SECOND

#define MULTIPLIER 10
// These numbers are from
// https://github.com/YerevaNN/Cooja-Automation-ML/blob/main/case_study_rpl/firmware/battery_client.c
#define CPU_CURRENT_MA 1.8 * MULTIPLIER
#define LPM_CURRENT_MA 0.0545 * MULTIPLIER
#define DEEP_LPM_CURRENT_MA 0.0135 * MULTIPLIER
#define RADIO_LISTEN_CURRENT_MA 20.0 * MULTIPLIER
#define RADIO_TRANSMIT_CURRENT_MA 17.4 * MULTIPLIER

/// -------------------- METRICS LOG -----------------------------------

#define DODAG_LOG                                                              \
  "DODAG: instance=%u version=%u rank=%u grounded=%u role=%s dag_id=%s "       \
  "preferred_parent=%s\n"
#define DODAG_NOT_JOIN "DODAG: not joined\n"

#define METRICS_LOG                                                            \
  "ENERGEST: CPU=%lu LPM=%lu DEEP_LPM=%lu LISTEN=%lu "                         \
  "TRANSMIT=%lu OFF=%lu TOTAL=%lu ENERGY_COMP=%lumA HOP_COUNT=%d "            \
  "ETX=%u.%02u\n"

#define LATENCY_LOG                                                            \
  "LATENCY: seqno=%" PRIu32 " rtt_ticks=%" PRIu32 " rtt_ms=%" PRIu32 "\n"
/// -------------------- METRICS LOG END -------------------------------

static unsigned long prev_cpu_tick = 0;
static clock_time_t prev_tick;
int hop_count = 0;

PROCESS(metrics_process, "Metrics process");

unsigned get_etx(void) {
  if (curr_instance.used) {
    rpl_parent_t *parent = curr_instance.dag.preferred_parent;
    if (parent != NULL) {
      const struct link_stats *stats = rpl_neighbor_get_link_stats(parent);
      if (stats != NULL) {
        uint16_t etx_x100 = (stats->etx * 100) / LINK_STATS_ETX_DIVISOR;
        return etx_x100;
      }
    }
  }
  return -1;
}

void metrics_print_dodag(void) {
  if (!curr_instance.used) {
    printf(DODAG_NOT_JOIN);
    return;
  }

  char dag_id[40];
  char preferred_parent[40] = "None";
  format_ipaddr(&curr_instance.dag.dag_id, dag_id, sizeof(dag_id));
  if (curr_instance.dag.preferred_parent != NULL) {
    format_ipaddr(rpl_neighbor_get_ipaddr(curr_instance.dag.preferred_parent),
                  preferred_parent, sizeof(preferred_parent));
  }
  printf(DODAG_LOG, curr_instance.instance_id, curr_instance.dag.version,
         curr_instance.dag.rank, curr_instance.dag.grounded,
         rpl_dag_root_is_root() ? "root" : "node", dag_id, preferred_parent);
}

uint32_t metrics_get_timestamp(void) { return (uint32_t)clock_time(); }

void metrics_log_latency(uint32_t seqno, uint32_t received_tick,
                         uint32_t sent_tick) {
  uint32_t rtt_tick = received_tick - sent_tick;
  uint32_t rtt_ms = (rtt_tick * 1000UL) / CLOCK_SECOND;

  printf(LATENCY_LOG, seqno, rtt_tick, rtt_ms);
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

  prev_tick = 0;
  prev_cpu_tick = 0;

  etimer_set(&metrics_timer, METRICS_PERIOD);

  // Periodic process: prints ETX, Energest, CPU util and Tx power
  while (1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&metrics_timer));

    energest_flush();
    unsigned long cpu = energest_type_time(ENERGEST_TYPE_CPU);
    unsigned long lpm = energest_type_time(ENERGEST_TYPE_LPM);
    unsigned long deep_lpm = energest_type_time(ENERGEST_TYPE_DEEP_LPM);
    unsigned long listen = energest_type_time(ENERGEST_TYPE_LISTEN);
    unsigned long transmit = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    unsigned long total = ENERGEST_GET_TOTAL_TIME();
    unsigned long off =
        total > (listen + transmit) ? total - listen - transmit : 0;

    double energy_comp =
        (cpu * CPU_CURRENT_MA + lpm * LPM_CURRENT_MA +
         deep_lpm * DEEP_LPM_CURRENT_MA + listen * RADIO_LISTEN_CURRENT_MA +
         transmit * RADIO_TRANSMIT_CURRENT_MA) /
        ENERGEST_SECOND;
    unsigned long energy_comp_mA = (unsigned long)energy_comp;

    unsigned etx = get_etx();
    unsigned etx_int = -1, etx_frac = 0;
    if (etx != -1) {
      etx_int = etx / 100;
      etx_frac = etx % 100;
    }


    printf(METRICS_LOG, cpu, lpm, deep_lpm, listen, transmit, off, total,
           energy_comp_mA, hop_count, etx_int, etx_frac);

    metrics_print_dodag();
    etimer_reset(&metrics_timer);
  }

  PROCESS_END();
}

void metrics_start(void) { process_start(&metrics_process, NULL); }
