#include "utils.h"
#include <stdio.h>

char *format_ipaddr(const uip_ipaddr_t *addr, char *buf, size_t buflen) {
  snprintf(
      buf, buflen,
      "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
      addr->u8[0], addr->u8[1], addr->u8[2], addr->u8[3], addr->u8[4],
      addr->u8[5], addr->u8[6], addr->u8[7], addr->u8[8], addr->u8[9],
      addr->u8[10], addr->u8[11], addr->u8[12], addr->u8[13], addr->u8[14],
      addr->u8[15]);
  return buf;
}
