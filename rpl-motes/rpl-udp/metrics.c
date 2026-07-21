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

#include "contiki.h"
#include "sys/energest.h"
#include "sys/rtimer.h"
#include "sys/clock.h"
#include "net/netstack.h"
#include "net/linkaddr.h"
#include "net/link-stats.h"
#include "net/ipv6/uip.h"
#include "net/routing/rpl-lite/rpl.h"
#include "net/routing/rpl-lite/rpl-dag-root.h"
#include "metrics.h"

#include <stdio.h>

/*---------------------------------------------------------------------*/
/* Internal state */
static unsigned long last_cpu_ticks = 0;
static clock_time_t last_total_ticks;
static rtimer_clock_t compute_start_ticks;

PROCESS(metrics_process, "Metrics process");

/*---------------------------------------------------------------------*/
/* ETX to preferred parent */
void
metrics_print_etx(void)
{
  if(curr_instance.used) {
    rpl_parent_t *parent = curr_instance.dag.preferred_parent;
    if(parent != NULL) {
      const struct link_stats *stats = rpl_neighbor_get_link_stats(parent);
      if(stats != NULL) {
        uint16_t etx_x10 = (stats->etx * 10) / LINK_STATS_ETX_DIVISOR;
        printf("ETX: %u.%u\n", etx_x10 / 10, etx_x10 % 10);
        return;
      }
    }
  }
  printf("ETX: no preferred parent\n");
}
/*---------------------------------------------------------------------*/
/* Helper: print an IPv6 address in the usual colon-hex form */
static void
print_ipaddr(const uip_ipaddr_t *addr)
{
  int i;
  for(i = 0; i < 16; i += 2) {
    printf("%02x%02x", addr->u8[i], addr->u8[i + 1]);
    if(i < 14) {
      printf(":");
    }
  }
}
/*---------------------------------------------------------------------*/
/* DODAG information: instance id, DAG ID, version, rank, grounded flag,
 * role, and preferred parent. */
void
metrics_print_dodag(void)
{
  if(!curr_instance.used) {
    printf("DODAG: not joined\n");
    return;
  }

  printf("DODAG: instance=%u version=%u rank=%u grounded=%u role=%s dag_id=",
         curr_instance.instance_id,
         curr_instance.dag.version,
         curr_instance.dag.rank,
         curr_instance.dag.grounded,
         rpl_dag_root_is_root() ? "root" : "node");
  print_ipaddr(&curr_instance.dag.dag_id);

  if(curr_instance.dag.preferred_parent != NULL) {
    printf(" preferred_parent=");
    print_ipaddr(rpl_neighbor_get_ipaddr(curr_instance.dag.preferred_parent));
  } else {
    printf(" preferred_parent=none");
  }
  printf("\n");
}
/*---------------------------------------------------------------------*/
/* Energy consumption via Energest */
void
metrics_print_energest(void)
{
  energest_flush();
  printf("ENERGEST: CPU %lu LPM %lu TX %lu RX %lu (ticks, %lu ticks/sec)\n",
         (unsigned long)energest_type_time(ENERGEST_TYPE_CPU),
         (unsigned long)energest_type_time(ENERGEST_TYPE_LPM),
         (unsigned long)energest_type_time(ENERGEST_TYPE_TRANSMIT),
         (unsigned long)energest_type_time(ENERGEST_TYPE_LISTEN),
         (unsigned long)ENERGEST_SECOND);
}
/*---------------------------------------------------------------------*/
/* CPU utilization (%) since the last call */
void
metrics_print_cpu_util(void)
{
  unsigned long cpu_ticks;
  clock_time_t now_ticks;
  unsigned long cpu_delta;
  unsigned long total_delta;

  energest_flush();
  cpu_ticks = (unsigned long)energest_type_time(ENERGEST_TYPE_CPU);
  now_ticks = clock_time();

  /* First call: nothing to compare against yet, just seed the state. */
  if(last_total_ticks == 0) {
    last_cpu_ticks = cpu_ticks;
    last_total_ticks = now_ticks;
    printf("CPU_UTIL: n/a (first sample)\n");
    return;
  }

  cpu_delta = cpu_ticks - last_cpu_ticks;
  total_delta = (unsigned long)now_ticks - (unsigned long)last_total_ticks;

  if(total_delta > 0) {
    unsigned long permille = (1000UL * cpu_delta) / total_delta;
    printf("CPU_UTIL: %lu.%lu%%\n", permille / 10, permille % 10);
  } else {
    printf("CPU_UTIL: n/a (no elapsed time)\n");
  }

  last_cpu_ticks = cpu_ticks;
  last_total_ticks = now_ticks;
}
/*---------------------------------------------------------------------*/
/* Tx power */
void
metrics_print_txpower(void)
{
  radio_value_t txpower;
  radio_result_t res = NETSTACK_RADIO.get_value(RADIO_PARAM_TXPOWER, &txpower);
  if(res == RADIO_RESULT_OK) {
    printf("TX_POWER: %d\n", (int)txpower);
  } else {
    printf("TX_POWER: unavailable\n");
  }
}
/*---------------------------------------------------------------------*/
/* Latency: sender-side helper to get a timestamp to embed in the packet */
uint32_t
metrics_get_timestamp(void)
{
  return (uint32_t)clock_time();
}
/*---------------------------------------------------------------------*/
/* Latency: receiver-side, call with the seqno/timestamp read from the
 * packet payload */
void
metrics_log_latency(uint16_t seqno, uint32_t sent_timestamp)
{
  uint32_t now = (uint32_t)clock_time();
  uint32_t latency_ticks = now - sent_timestamp;
  uint32_t latency_ms = (latency_ticks * 1000UL) / CLOCK_SECOND;
  printf("LATENCY: seqno=%u ticks=%lu ms=%lu\n",
         seqno, (unsigned long)latency_ticks, (unsigned long)latency_ms);
}
/*---------------------------------------------------------------------*/
/* Hop count: call inside the packet-receive callback, while uip_buf still
 * holds the packet just received. */
void
metrics_print_hop_count(uint8_t initial_ttl)
{
  uint8_t ttl = UIP_IP_BUF->ttl;
  if(ttl <= initial_ttl) {
    printf("HOP_COUNT: %u\n", (unsigned)(initial_ttl - ttl));
  } else {
    printf("HOP_COUNT: n/a (ttl=%u > initial_ttl=%u)\n", ttl, initial_ttl);
  }
}
/*---------------------------------------------------------------------*/
/* Computing time: simple rtimer-based stopwatch, one measurement at a
 * time (not nestable). */
void
metrics_time_start(void)
{
  compute_start_ticks = RTIMER_NOW();
}

void
metrics_time_end(const char *label)
{
  rtimer_clock_t elapsed = RTIMER_NOW() - compute_start_ticks;
  uint32_t elapsed_us = (uint32_t)(((uint64_t)elapsed * 1000000) / RTIMER_SECOND);
  printf("COMPUTE_TIME: %s ticks=%lu us=%lu\n",
         label, (unsigned long)elapsed, (unsigned long)elapsed_us);
}
/*---------------------------------------------------------------------*/
/* Periodic process: prints ETX, Energest, CPU util and Tx power every
 * METRICS_PERIOD seconds. Per-packet metrics (latency, hop count) are
 * NOT printed here - call metrics_log_latency() / metrics_print_hop_count()
 * directly from your send/receive callbacks. */
PROCESS_THREAD(metrics_process, ev, data)
{
  static struct etimer et;

  PROCESS_BEGIN();

  last_total_ticks = 0;
  last_cpu_ticks = 0;

  etimer_set(&et, CLOCK_SECOND * METRICS_PERIOD);

  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&et));

    metrics_print_etx();
    metrics_print_dodag();
    metrics_print_energest();
    metrics_print_cpu_util();
    metrics_print_txpower();

    etimer_reset(&et);
  }

  PROCESS_END();
}
/*---------------------------------------------------------------------*/
void
metrics_start(void)
{
  process_start(&metrics_process, NULL);
}
/*---------------------------------------------------------------------*/
