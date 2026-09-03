CONTIKI_PROJECT = udp-server

BUFFER_SIZE ?= 16
DAO_ACK ?= 1
RPL_OF ?= 1
NETWORK_SIZE ?= 30

ifeq ($(RPL_OF),0)
RPL_SUPPORTED_OF ?= rpl_of0
else ifeq ($(RPL_OF),2)
RPL_SUPPORTED_OF ?= rpl_mlof
else
RPL_SUPPORTED_OF ?= rpl_mrhof
endif

all: $(CONTIKI_PROJECT)

# NETWORK_SIZE sizes the root source-routing pool only (RPL-Lite is non-storing,
# so uip-ds6-route's UIP_MAX_ROUTES is already 0). Name it explicitly.
CFLAGS += -DQUEUEBUF_CONF_NUM=$(BUFFER_SIZE) -DRPL_CONF_WITH_DAO_ACK=$(DAO_ACK) -DRPL_OF=$(RPL_OF) -DRPL_CONF_SUPPORTED_OFS="{&$(RPL_SUPPORTED_OF)}" -DUIP_SR_CONF_LINK_NUM=$(NETWORK_SIZE) -DNETSTACK_MAX_ROUTE_ENTRIES=0

# z1 has a hardware AES engine (cc2420); drop the unused software aes-128.c
# (a 176 B round-key table that -fdata-sections cannot strip on msp430x).
MODULES_SOURCES_EXCLUDES += aes-128.c

CONTIKI = ../../contiki-ng
include $(CONTIKI)/Makefile.include
