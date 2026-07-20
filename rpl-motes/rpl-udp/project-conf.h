#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define ENERGEST_CONF_ON 1

#ifdef CONTIKI_TARGET_SKY
/* Save some RAM and ROM */
#define QUEUEBUF_CONF_NUM              4
#define UIP_CONF_BUFFER_SIZE         140
#define BORDER_ROUTER_CONF_WEBSERVER   0
#endif

#endif /* PROJECT_CONF_H_ */
