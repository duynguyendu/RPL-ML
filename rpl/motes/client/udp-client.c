#include "metrics.h"
#include "net/ipv6/simple-udp.h"
#include "net/netstack.h"
#include "net/routing/routing.h"
#include "random.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sys/log.h"
#define LOG_MODULE "App"
#define LOG_LEVEL LOG_LEVEL_INFO

#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678

#ifndef RAMP_UP_DURATION
#define RAMP_UP_DURATION 60
#endif

#ifndef SEND_RATE
#define SEND_RATE 30
#endif

#define SEND_TICK (SEND_RATE * CLOCK_SECOND)

#define MAX_PENDING 20

static struct simple_udp_connection udp_conn;
int hop_count = -1;

#if GATHER_METRICS
static clock_time_t send_times[MAX_PENDING];
#endif

/*---------------------------------------------------------------------------*/
PROCESS(udp_client_process, "UDP client");
AUTOSTART_PROCESSES(&udp_client_process);
/*---------------------------------------------------------------------------*/
static void udp_rx_callback(struct simple_udp_connection *c,
                            const uip_ipaddr_t *sender_addr,
                            uint16_t sender_port,
                            const uip_ipaddr_t *receiver_addr,
                            uint16_t receiver_port, const uint8_t *data,
                            uint16_t datalen) {

  hop_count = get_hop_count(UIP_TTL);
  printf("HOP_COUNT=%u Received response '%.*s' from ", hop_count, datalen, (char *)data);
  LOG_INFO_6ADDR(sender_addr);
  printf("\n");

#if GATHER_METRICS
  uint32_t received_tick = metrics_get_timestamp();
  // Extract seqno from the response data
  uint32_t seqno = atoi((char *)data);
  metrics_log_latency(seqno, received_tick, send_times[seqno % MAX_PENDING]);
#endif
}

PROCESS_THREAD(udp_client_process, ev, data) {
  static struct etimer periodic_timer;
  static char str[PACKET_SIZE];
  uip_ipaddr_t dest_ipaddr;
  static uint32_t tx_count;

  PROCESS_BEGIN();

  /* Initialize UDP connection */
  simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL, UDP_SERVER_PORT,
                      udp_rx_callback);

  // Wait till DODAG stablise
  etimer_set(&periodic_timer, RAMP_UP_DURATION * CLOCK_SECOND);
  PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

#if GATHER_METRICS
  metrics_start();
#endif

  etimer_set(&periodic_timer, (random_rand() % SEND_RATE) * CLOCK_SECOND);
  while (1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

    NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr);
    printf("Sending request '%" PRIu32 "' to ", tx_count);
    LOG_INFO_6ADDR(&dest_ipaddr);
    printf("\n");

    snprintf(str, sizeof(str), "%" PRIu32 "", tx_count);
#if GATHER_METRICS
    send_times[tx_count % MAX_PENDING] = metrics_get_timestamp();
#endif

    simple_udp_sendto(&udp_conn, str, PACKET_SIZE, &dest_ipaddr);
    tx_count++;

    // TODO: add 20% of jitter
    etimer_set(&periodic_timer, SEND_TICK);
  }

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
