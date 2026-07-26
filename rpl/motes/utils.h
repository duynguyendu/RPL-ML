#pragma once

#include "net/ipv6/uip.h"
#include "sys/energest.h"

void print_ipaddr(const uip_ipaddr_t *addr);

unsigned long ticks_to_seconds(uint64_t ticks) {
  return (unsigned long)(ticks / ENERGEST_SECOND);
}
