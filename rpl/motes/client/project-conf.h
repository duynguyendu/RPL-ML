#include "../proj-logging-conf.h"

#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#if GATHER_METRICS
#define ENERGEST_CONF_ON 1
#endif

#ifndef PACKET_SIZE
#define PACKET_SIZE 32
#endif

#endif /* PROJECT_CONF_H_ */
