#pragma once

#include "net/ipv6/uip.h"

char *format_ipaddr(const uip_ipaddr_t *addr, char *buf, size_t buflen);
