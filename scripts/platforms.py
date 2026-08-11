from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import config


@dataclass(frozen=True)
class PlatformSpec:
    base_dir: Path
    name: str
    mote_type: str
    target: str
    server_file_name: str = "udp-server"
    client_file_name: str = "udp-client"
    overload_client_file_name: str = "overload-client"
    interfaces: List[str] = field(default_factory=list)

    def server_base_dir(self):
        return self.base_dir / "server"

    def client_base_dir(self):
        return self.base_dir / "client"

    def server_binary_name(self):
        return f"{self.server_file_name}.{self.target}"

    def server_firmware_path(self):
        return (
            self.server_base_dir() / f"build/{self.target}/{self.server_binary_name()}"
        ).as_posix()

    def server_source_path(self):
        return (self.server_base_dir() / f"{self.server_file_name}.c").as_posix()

    def client_binary_name(self):
        return f"{self.client_file_name}.{self.target}"

    def client_firmware_path(self):
        return (
            self.client_base_dir() / f"build/{self.target}/{self.client_binary_name()}"
        ).as_posix()

    def client_source_path(self):
        return (self.client_base_dir() / f"{self.client_file_name}.c").as_posix()

    def overload_client_binary_name(self):
        return f"{self.overload_client_file_name}.{self.target}"

    def overload_client_source_name(self):
        return f"{self.overload_client_file_name}.c"


PLATFORMS: Dict[str, PlatformSpec] = {
    "z1": PlatformSpec(
        base_dir=Path(config.base_mote_dir).resolve(),
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
    "cooja": PlatformSpec(
        base_dir=Path(config.base_mote_dir).resolve(),
        name="cooja",
        mote_type="org.contikios.cooja.contikimote.ContikiMoteType",
        target="cooja",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.Battery",
            "org.contikios.cooja.contikimote.interfaces.ContikiVib",
            "org.contikios.cooja.contikimote.interfaces.ContikiMoteID",
            "org.contikios.cooja.contikimote.interfaces.ContikiRS232",
            "org.contikios.cooja.contikimote.interfaces.ContikiBeeper",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.contikimote.interfaces.ContikiRadio",
            "org.contikios.cooja.contikimote.interfaces.ContikiButton",
            "org.contikios.cooja.contikimote.interfaces.ContikiPIR",
            "org.contikios.cooja.contikimote.interfaces.ContikiClock",
            "org.contikios.cooja.contikimote.interfaces.ContikiLED",
            "org.contikios.cooja.contikimote.interfaces.ContikiCFS",
            "org.contikios.cooja.contikimote.interfaces.ContikiEEPROM",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
        ],
    ),
    # Preview entries below; validate in your environment and add other interfaces as needed before production sweeps.
    "sky": PlatformSpec(
        base_dir=Path(config.base_mote_dir).resolve(),
        name="sky",
        mote_type="org.contikios.cooja.mspmote.SkyMoteType",
        target="sky",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.SkyButton",
            "org.contikios.cooja.mspmote.interfaces.SkyFlash",
            "org.contikios.cooja.mspmote.interfaces.SkyCoffeeFilesystem",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput",
            "org.contikios.cooja.mspmote.interfaces.SkyTemperature",
        ],
    ),
    "wismote": PlatformSpec(
        base_dir=Path(config.base_mote_dir).resolve(),
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
