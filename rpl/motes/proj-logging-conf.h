#pragma once

#define RPL_CONF_OF_OCP RPL_OF

#define LOG_CONF_LEVEL_RPL LOG_LEVEL_WARN

/*
#define RPL_CONF_MULTIPLE_METRICS 0
*/
#define RPL_CONF_MC_MAX_METRICS 3

#define TSCH_LOG_CONF_PER_SLOT 0
#define QUEUEBUF_CONF_STATS 0

/* --- Static-RAM trims (Tier A) --- shared by udp-server and udp-client --- */

/* 6LoWPAN reassembly pools (net/ipv6/sicslowpan.c). Fragmentation stays on
 * (SICSLOWPAN_CONF_FRAG), but the default 12 fragment buffers / 2 reassembly
 * contexts are far larger than this workload (<=64 B UDP payloads) needs. */
#define SICSLOWPAN_CONF_FRAGMENT_BUFFERS 3   /* was 12; ~114 B each */
#define SICSLOWPAN_CONF_REASS_CONTEXTS  1    /* was 2;  ~172 B each */

/* CSMA duplicate-frame history (net/mac/mac-sequence.c). */
#define NETSTACK_CONF_MAC_SEQNO_HISTORY 8    /* was 16; 14 B each */

/* Serial-line command *input* buffer (os/dev/serial-line.c); must be a power
 * of two. Does not affect mote->Cooja log output. */
#define SERIAL_LINE_CONF_BUFSIZE 32          /* was 128; ~2 B/unit x2 */

/* uIP DS6 table units: one DODAG, one prefix, one default route. */
#define UIP_CONF_DS6_ADDR_NBU   1            /* was 2 */
#define UIP_CONF_DS6_PREFIX_NBU 1            /* was 2 */
#define UIP_CONF_DS6_DEFRT_NBU  1            /* was 2; non-root needs >= 1 */

/* Per-neighbor tx/rx/ack/drop counters (+16 B/neighbor). Only the client's
 * metrics.c reads these; the server never does and defines no GATHER_METRICS. */
#if defined(GATHER_METRICS) && GATHER_METRICS
#define LINK_STATS_CONF_PACKET_COUNTERS 1
#endif
