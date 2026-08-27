#include "../proj-logging-conf.h"

#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define TSCH_SCHEDULE_CONF_MAX_LINKS 16
#define TSCH_LOG_CONF_PER_SLOT 0

#if GATHER_METRICS
#define ENERGEST_CONF_ON 1
#endif

#ifndef PACKET_SIZE
#define PACKET_SIZE 32
#endif

#ifdef CONTIKI_TARGET_SKY
/* Save some RAM and ROM */

#ifndef QUEUEBUF_CONF_NUM
#define QUEUEBUF_CONF_NUM 8
#endif /* QUEUEBUF_CONF_NUM */

#define UIP_CONF_BUFFER_SIZE 140
#define BORDER_ROUTER_CONF_WEBSERVER 0
#endif /* CONTIKI_TARGET_SKY */

#endif /* PROJECT_CONF_H_ */
