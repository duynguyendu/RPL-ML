#!/usr/bin/env python3
"""
Generalized CSC generator: topology JSON → Cooja .csc with pluggable platform.

Backwards-compatible defaults follow the existing z1 setup.

JSON expectations:
- keys: version, topology_id, seed, radio {tx_range, interference_range, success_tx, success_rx},
  timing.duration_s, motes: list of {id, role (server|client), x, y[, z]}.

Compile model:
- We build firmware via `make` using the repository root as project dir and set BUILD_DIR when provided.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import config

try:
    # When executed as module (python -m ...)
    from .platforms import PlatformSpec, get_platform
except Exception:  # pragma: no cover
    # Allow running as a standalone script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from platforms import PlatformSpec, get_platform


def _join_interfaces(interfaces: List[str]) -> str:
    return "\n".join(
        [f"      <moteinterface>{iface}</moteinterface>" for iface in interfaces]
    )


def convert_rpl_of_to_int(rpl_of):
    if rpl_of == "of0":
        return 0
    elif rpl_of == "mhrof":
        return 1
    # default to mhrof
    return 1


def make_header(
    title: str,
    seed: int,
    radio: Dict[str, Any],
    platform_spec: PlatformSpec,
) -> str:
    # Original repository root that contains src/ and Makefile
    target = platform_spec.target

    parameters = f"TARGET={target} BUFFER_SIZE={config.buffer_size} SEND_RATE={config.send_rate} DAO_ACK={config.with_dao_ack} RAMP_UP_DURATION={config.ramp_up_duration} PACKET_SIZE={config.packet_size} RPL_OF={convert_rpl_of_to_int(config.rpl_of)}"

    server_cmd = f"$(MAKE) -C {platform_spec.server_base_dir()} -j$(CPUS) {platform_spec.server_binary_name()} {parameters}"
    client_cmd = f"$(MAKE) -C {platform_spec.client_base_dir()} -j$(CPUS) {platform_spec.client_binary_name()} {parameters} GATHER_METRICS={config.gather_metrics}"

    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<simconf>
  <simulation>
    <title>{title}</title>
    <randomseed>{seed}</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>{radio["tx_range"]}</transmitting_range>
      <interference_range>{radio["interference_range"]}</interference_range>
      <success_ratio_tx>{radio.get("success_tx", 1.0)}</success_ratio_tx>
      <success_ratio_rx>{radio.get("success_rx", 1.0)}</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>

    <motetype>
      {platform_spec.mote_type}
      <identifier>{platform_spec.name}_server</identifier>
      <description>RPL Server (Root) - {platform_spec.name.upper()}</description>
      <source>{platform_spec.server_source_path()}</source>
      <commands>{server_cmd}</commands>
      <firmware>{platform_spec.server_firmware_path()}</firmware>
{_join_interfaces(platform_spec.interfaces)}
    </motetype>

    <motetype>
      {platform_spec.mote_type}
      <identifier>{platform_spec.name}_client</identifier>
      <description>RPL Client - {platform_spec.name.upper()}</description>
      <source>{platform_spec.client_source_path()}</source>
      <commands>{client_cmd}</commands>
      <firmware>{platform_spec.client_firmware_path()}</firmware>
{_join_interfaces(platform_spec.interfaces)}
    </motetype>

"""


#     <motetype>
#       {platform_spec.mote_type}
#       <identifier>{platform_spec.name}_overload_client</identifier>
#       <description>RPL Overload Client - {platform_spec.name.upper()}</description>
#       <source>{overload_client_source}</source>
#       <commands>{overload_client_cmd}</commands>
#       <firmware>{overload_client_fw}</firmware>
# {_join_interfaces(platform_spec.interfaces)}
#     </motetype>


def make_footer(timeout_ms: int, server_id: int) -> str:
    script_js = """
// Headless logging for parser compatibility
TIMEOUT(__TIMEOUT__, log.testOK());
while (true) {
  YIELD();
  log.log(time + ":" + id + ":" + msg + "\\n");
}
"""
    script_js = script_js.replace("__TIMEOUT__", str(timeout_ms))

    return f"""
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script><![CDATA[
{script_js}
      ]]></script>
      <active>true</active>
    </plugin_config>
    <bounds x="0" y="0" height="100" width="100" />
  </plugin>

  <plugin>
    org.contikios.cooja.plugins.Visualizer
    <plugin_config>
      <moterelations>true</moterelations>
      <skin>org.contikios.cooja.plugins.skins.IDVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.GridVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.TrafficVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.UDGMVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.MoteTypeVisualizerSkin</skin>
      <viewport>2.0 0.0 0.0 2.0 204.0 87.0</viewport>
    </plugin_config>
    <bounds x="899" y="20" height="512" width="512" />
  </plugin>
</simconf>"""


def mote_xml(
    mote_id: int, x: float, y: float, motetype: str, z: float | None = None
) -> str:
    if z is None:
        z = 0.0
    return f"""  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>{x}</x>
      <y>{y}</y>
      <z>{z}</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>{mote_id}</id>
    </interface_config>
    <motetype_identifier>{motetype}</motetype_identifier>
  </mote>"""


def generate_csc_from_dict(topo: Dict[str, Any]) -> str:
    platform_spec = get_platform(config.platform)
    radio = topo["radio"]

    build_dir = platform_spec.server_base_dir() / "build"
    if build_dir.exists() and build_dir.is_dir():
        shutil.rmtree(build_dir)
    build_dir = platform_spec.client_base_dir() / "build"
    if build_dir.exists() and build_dir.is_dir():
        shutil.rmtree(build_dir)

    header = make_header(
        title=topo.get("topology_id", topo.get("title", "cooja_run")),
        seed=topo.get("seed", 123456),
        radio=radio,
        platform_spec=platform_spec,
    )

    motes_xml = []
    for m in topo["motes"]:
        role = str(m.get("role", "client")).lower()
        motetype = f"{platform_spec.name}_{role}"
        motes_xml.append(
            mote_xml(
                int(m["id"]),
                float(m["x"]),
                float(m["y"]),
                motetype,
                float(m.get("z", 0.0)),
            )
        )

    duration_s = int(topo.get("timing", {}).get("duration_s", 180))

    # Determine server id
    server_id = None
    for m in topo.get("motes", []):
        if str(m.get("role", "")).lower() == "server":
            server_id = int(m["id"])
            break
    if server_id is None:
        server_id = 1

    footer = make_footer(timeout_ms=duration_s * 1000, server_id=server_id)
    return header + "\n".join(motes_xml) + footer


def generate_csc(
    topology_json: str,
    out_csc: str,
    base_dir: str,
) -> None:
    topo = json.loads(Path(topology_json).read_text())
    csc = generate_csc_from_dict(topo)

    Path(out_csc).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csc).write_text(csc)
    print(f"Wrote {out_csc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate CSC from topology JSON (pluggable platform)"
    )
    ap.add_argument("topology_json", type=str)
    ap.add_argument("out_csc", type=str)
    ap.add_argument(
        "--platform", type=str, default="sky", help="Platform name (default: sky)"
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory for mote directory",
    )
    args = ap.parse_args()

    generate_csc(
        topology_json=args.topology_json,
        out_csc=args.out_csc,
        base_dir=args.base_dir,
        platform=args.platform,
    )
