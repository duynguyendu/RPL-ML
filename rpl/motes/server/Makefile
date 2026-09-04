CONTIKI_PROJECT = udp-server

BUFFER_SIZE ?= 16
DAO_ACK ?= 1
RPL_OF ?= 1
NETWORK_SIZE ?= 30

# Per-module log levels (Contiki-NG LOG_LEVEL_{NONE,ERR,WARN,INFO,DBG});
# passed in by scripts/csc_generator.py, overridable on the make command line.
LOG_CONF_LEVEL_RPL ?= LOG_LEVEL_NONE
LOG_CONF_LEVEL_IPV6 ?= LOG_LEVEL_NONE
LOG_CONF_LEVEL_6LOWPAN ?= LOG_LEVEL_NONE
LOG_CONF_LEVEL_MAC ?= LOG_LEVEL_NONE
LOG_CONF_LEVEL_FRAMER ?= LOG_LEVEL_NONE

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
CFLAGS += -DLOG_CONF_LEVEL_RPL=$(LOG_CONF_LEVEL_RPL) -DLOG_CONF_LEVEL_IPV6=$(LOG_CONF_LEVEL_IPV6) -DLOG_CONF_LEVEL_6LOWPAN=$(LOG_CONF_LEVEL_6LOWPAN) -DLOG_CONF_LEVEL_MAC=$(LOG_CONF_LEVEL_MAC) -DLOG_CONF_LEVEL_FRAMER=$(LOG_CONF_LEVEL_FRAMER)

# z1 has a hardware AES engine (cc2420); drop the unused software aes-128.c
# (a 176 B round-key table that -fdata-sections cannot strip on msp430x).
ifeq ($(TARGET),z1)
MODULES_SOURCES_EXCLUDES += aes-128.c
endif

CONTIKI = ../../contiki-ng
include $(CONTIKI)/Makefile.include
