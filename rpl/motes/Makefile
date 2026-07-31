CONTIKI_PROJECT = udp-client overload-client udp-server
all: $(CONTIKI_PROJECT)

PROJECT_SOURCEFILES += metrics.c
PROJECT_SOURCEFILES += utils.c

udp-client: CFLAGS += -DBUILD_CLIENT=1
overload-client: CFLAGS += -DBUILD_CLIENT=1
udp-server: CFLAGS += -DBUILD_SERVER=1

CONTIKI = ../contiki-ng
include $(CONTIKI)/Makefile.include
