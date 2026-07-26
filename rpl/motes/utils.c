#include "utils.h"
#include <stdio.h>

void print_ipaddr(const uip_ipaddr_t *addr) {
  int i;
  for (i = 0; i < 16; i += 2) {
    printf("%02x%02x", addr->u8[i], addr->u8[i + 1]);
    if (i < 14) {
      printf(":");
    }
  }
}
