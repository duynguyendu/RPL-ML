#include "../proj-logging-conf.h"

#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define TSCH_LOG_CONF_PER_SLOT 0

#define NETSTACK_MAX_ROUTE_ENTRIES 60
#define RPL_CONF_OF_OCP RPL_OF

#ifdef CONTIKI_TARGET_SKY
/* Save some RAM and ROM */

#ifndef QUEUEBUF_CONF_NUM
#define QUEUEBUF_CONF_NUM 8
#endif /* QUEUEBUF_CONF_NUM */

#define UIP_CONF_BUFFER_SIZE 140
#define BORDER_ROUTER_CONF_WEBSERVER 0
#endif /* CONTIKI_TARGET_SKY */

#endif /* PROJECT_CONF_H_ */
