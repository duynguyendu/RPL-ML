#include "../proj-logging-conf.h"

#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define LOG_CONF_LEVEL_MAC LOG_LEVEL_WARN
#define LOG_CONF_LEVEL_6LOWPAN LOG_LEVEL_WARN
#define LOG_CONF_LEVEL_IPV6 LOG_LEVEL_WARN

#if GATHER_METRICS
#define ENERGEST_CONF_ON 1
#endif

#ifndef PACKET_SIZE
#define PACKET_SIZE 32
#endif

#endif /* PROJECT_CONF_H_ */
