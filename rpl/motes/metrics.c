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
#include "contiki.h"
#include "net/ipv6/uip.h"
#include "net/link-stats.h"
#include "net/linkaddr.h"
#include "net/netstack.h"
#include "net/routing/rpl-lite/rpl-dag-root.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/clock.h"
#include "sys/energest.h"
#include "sys/rtimer.h"
#include "utils.h"

#include <stdio.h>

/*---------------------------------------------------------------------*/
/* Internal state */
static unsigned long prev_cpu_tick = 0;
static clock_time_t prev_tick;
static rtimer_clock_t compute_start_ticks;

PROCESS(metrics_process, "Metrics process");

void metrics_print_etx(void) {
  if (curr_instance.used) {
    rpl_parent_t *parent = curr_instance.dag.preferred_parent;
    if (parent != NULL) {
      const struct link_stats *stats = rpl_neighbor_get_link_stats(parent);
      if (stats != NULL) {
        uint16_t etx_x10 = (stats->etx * 10) / LINK_STATS_ETX_DIVISOR;
        // TODO: maybe print parents as well
        printf("ETX: %u.%u\n", etx_x10 / 10, etx_x10 % 10);
        return;
      }
    }
  }
  printf("ETX: no preferred parent\n");
}

void metrics_print_dodag(void) {
  if (!curr_instance.used) {
    printf("DODAG: not joined\n");
    return;
  }

  printf("DODAG: instance=%u version=%u rank=%u grounded=%u role=%s dag_id=",
         curr_instance.instance_id, curr_instance.dag.version,
         curr_instance.dag.rank, curr_instance.dag.grounded,
         rpl_dag_root_is_root() ? "root" : "node");
  print_ipaddr(&curr_instance.dag.dag_id);

  if (curr_instance.dag.preferred_parent != NULL) {
    printf(" preferred_parent=");
    print_ipaddr(rpl_neighbor_get_ipaddr(curr_instance.dag.preferred_parent));
  } else {
    printf(" preferred_parent=none");
  }
  printf("\n");
}

void metrics_print_energest(void) {
  unsigned long cpu = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_CPU));
  unsigned long lpm = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_LPM));
  unsigned long deep_lpm =
      ticks_to_seconds(energest_type_time(ENERGEST_TYPE_DEEP_LPM));
  unsigned long listen =
      ticks_to_seconds(energest_type_time(ENERGEST_TYPE_LISTEN));
  unsigned long transmit =
      ticks_to_seconds(energest_type_time(ENERGEST_TYPE_TRANSMIT));
  unsigned long total = ticks_to_seconds(ENERGEST_GET_TOTAL_TIME());
  unsigned long off =
      total > (listen + transmit) ? total - listen - transmit : 0;

  printf("ENERGEST: CPU=%lus LPM=%lus DEEP_LPM=%lus LISTEN=%lus "
         "TRANSMIT=%lus OFF=%lus TOTAL=%lus\n",
         cpu, lpm, deep_lpm, listen, transmit, off, total);
}

void metrics_print_cpu_util(void) {
  unsigned long current_cpu_tick =
      (unsigned long)energest_type_time(ENERGEST_TYPE_CPU);
  clock_time_t current_tick = clock_time();

  /* First call: nothing to compare against yet, just seed the state. */
  if (prev_tick == 0) {
    prev_cpu_tick = current_cpu_tick;
    prev_tick = current_tick;
    printf("CPU_UTIL: n/a (first sample)\n");
    return;
  }

  unsigned long cpu_tick_delta = current_cpu_tick - prev_cpu_tick;
  unsigned long tick_delta =
      (unsigned long)current_tick - (unsigned long)prev_tick;

  if (tick_delta > 0) {
    // TODO: verify that this is cpu usage
    unsigned long percent = (100UL * cpu_tick_delta) / tick_delta;
    printf("CPU_UTIL: %lu.%lu%%\n", percent / 10, percent % 10);
  } else {
    printf("CPU_UTIL: n/a (no elapsed time)\n");
  }

  prev_cpu_tick = current_cpu_tick;
  prev_tick = current_tick;
}

void metrics_energest(void) {
  energest_flush();

  // TODO: calculate energy consumption
  // TODO: what should be the unit of the energy

  metrics_print_energest();
  metrics_print_cpu_util();
}

// TODO: double check this metrics
void metrics_print_txpower(void) {
  radio_value_t txpower;
  radio_result_t res = NETSTACK_RADIO.get_value(RADIO_PARAM_TXPOWER, &txpower);
  if (res == RADIO_RESULT_OK) {
    printf("TX_POWER: %d\n", (int)txpower);
  } else {
    printf("TX_POWER: unavailable\n");
  }
}

uint32_t metrics_get_timestamp(void) { return (uint32_t)clock_time(); }

// TODO: call this
void metrics_log_latency(uint16_t seqno, uint32_t sent_timestamp) {
  uint32_t now = metrics_get_timestamp();
  uint32_t latency_ticks = now - sent_timestamp;
  uint32_t latency_ms = (latency_ticks * 1000UL) / CLOCK_SECOND;
  printf("LATENCY: seqno=%u ticks=%lu ms=%lu\n", seqno,
         (unsigned long)latency_ticks, (unsigned long)latency_ms);
}

// TODO: what is ttl
void metrics_print_hop_count(uint8_t initial_ttl) {
  uint8_t ttl = UIP_IP_BUF->ttl;
  if (ttl <= initial_ttl) {
    printf("HOP_COUNT: %u\n", (unsigned)(initial_ttl - ttl));
  } else {
    printf("HOP_COUNT: n/a (ttl=%u > initial_ttl=%u)\n", ttl, initial_ttl);
  }
}

// TODO: probably move to RPL impl
/*---------------------------------------------------------------------*/
/* Computing time: simple rtimer-based stopwatch, one measurement at a
 * time (not nestable). Use to check how long to run the AI model */
void metrics_time_start(void) { compute_start_ticks = RTIMER_NOW(); }

void metrics_time_end(const char *label) {
  rtimer_clock_t elapsed = RTIMER_NOW() - compute_start_ticks;
  uint32_t elapsed_us =
      (uint32_t)(((uint64_t)elapsed * 1000000) / RTIMER_SECOND);
  printf("COMPUTE_TIME: %s ticks=%lu us=%lu\n", label, (unsigned long)elapsed,
         (unsigned long)elapsed_us);
}

PROCESS_THREAD(metrics_process, ev, data) {
  static struct etimer metrics_timer;

  PROCESS_BEGIN();

  prev_tick = 0;
  prev_cpu_tick = 0;

  etimer_set(&metrics_timer, CLOCK_SECOND * METRICS_PERIOD);

  // Periodic process: prints ETX, Energest, CPU util and Tx power
  while (1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&metrics_timer));

    metrics_print_etx();
    metrics_print_dodag();
    metrics_energest();
    metrics_print_cpu_util();
    metrics_print_txpower();

    etimer_reset(&metrics_timer);
  }

  PROCESS_END();
}

void metrics_start(void) { process_start(&metrics_process, NULL); }
