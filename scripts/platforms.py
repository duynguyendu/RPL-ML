from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    mote_type: str
    target: str
    server_file_name: str = "udp-server"
    client_file_name: str = "udp-client"
    overload_client_file_name: str = "overload-client"
    interfaces: List[str] = field(default_factory=list)

    def server_binary_name(self):
        return f"{self.server_file_name}.{self.target}"

    def server_source_name(self):
        return f"{self.server_file_name}.c"

    def client_binary_name(self):
        return f"{self.client_file_name}.{self.target}"

    def client_source_name(self):
        return f"{self.client_file_name}.c"

    def overload_client_binary_name(self):
        return f"{self.overload_client_file_name}.{self.target}"

    def overload_client_source_name(self):
        return f"{self.overload_client_file_name}.c"


PLATFORMS: Dict[str, PlatformSpec] = {
    "z1": PlatformSpec(
        name="z1",
        mote_type="org.contikios.cooja.mspmote.Z1MoteType",
        target="z1",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput",
        ],
    ),
    # Preview entries below; validate in your environment and add other interfaces as needed before production sweeps.
    "sky": PlatformSpec(
        name="sky",
        mote_type="org.contikios.cooja.mspmote.SkyMoteType",
        target="sky",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.SkyButton",
            "org.contikios.cooja.mspmote.interfaces.SkyFlash",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput",
            "org.contikios.cooja.mspmote.interfaces.SkyLED",
        ],
    ),
    "wismote": PlatformSpec(
        name="wismote",
        mote_type="org.contikios.cooja.mspmote.WismoteMoteType",
        target="wismote",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput",
        ],
    ),
}


def get_platform(name: str) -> PlatformSpec:
    key = (name or "z1").lower()
    if key not in PLATFORMS:
        raise ValueError(f"Unknown platform '{name}'. Available: {sorted(PLATFORMS)}")
    return PLATFORMS[key]
