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

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

try:
    # When executed as module (python -m ...)
    from .platforms import get_platform, PlatformSpec
except Exception:  # pragma: no cover
    # Allow running as a standalone script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from platforms import get_platform, PlatformSpec


def _join_interfaces(interfaces: List[str]) -> str:
    return "\n".join(
        [f"      <moteinterface>{iface}</moteinterface>" for iface in interfaces]
    )


def make_header(
    title: str,
    seed: int,
    radio: Dict[str, Any],
    platform: PlatformSpec,
    build_root: str | None = None,
    expected_nodes: int | None = None,
    base_dir: str | None = None,
) -> str:
    # Original repository root that contains src/ and Makefile
    base_dir = Path(base_dir).resolve()

    server_source = (base_dir / platform.server_source_name()).as_posix()
    client_source = (base_dir / platform.client_source_name()).as_posix()

    if build_root:
        build_root_path = Path(build_root)
        server_fw = (
            build_root_path / platform.target / platform.server_binary_name()
        ).as_posix()
        client_fw = (
            build_root_path / platform.target / platform.client_binary_name()
        ).as_posix()
        extra_vars = f" BUILD_DIR={build_root_path.as_posix()}"
    else:
        server_fw = (
            base_dir / f"build/{platform.target}/{platform.server_binary_name()}"
        ).as_posix()
        client_fw = (
            base_dir / f"build/{platform.target}/{platform.client_binary_name()}"
        ).as_posix()
        extra_vars = ""

    defines = ""
    if expected_nodes is not None and expected_nodes >= 0:
        defines = f" DEFINES+=EXPECTED_NODES={expected_nodes}"

    server_cmd = f"$(MAKE) -C {base_dir.as_posix()} -j$(CPUS){extra_vars} {defines} {platform.server_binary_name()} TARGET={platform.target}"
    client_cmd = f"$(MAKE) -C {base_dir.as_posix()} -j$(CPUS){extra_vars} {platform.client_binary_name()} TARGET={platform.target}"

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
      {platform.mote_type}
      <identifier>{platform.name}_server</identifier>
      <description>RPL Server (Root) - {platform.name.upper()}</description>
      <source>{server_source}</source>
      <commands>{server_cmd}</commands>
      <firmware>{server_fw}</firmware>
{_join_interfaces(platform.interfaces)}
    </motetype>

    <motetype>
      {platform.mote_type}
      <identifier>{platform.name}_client</identifier>
      <description>RPL Client - {platform.name.upper()}</description>
      <source>{client_source}</source>
      <commands>{client_cmd}</commands>
      <firmware>{client_fw}</firmware>
{_join_interfaces(platform.interfaces)}
    </motetype>
"""


def make_footer(timeout_ms: int, server_id: int) -> str:
    script_js = """
// Headless logging for parser compatibility
TIMEOUT(__TIMEOUT__, log.testOK());
var ROOT_ID = __ROOT__;
function formatTime(microseconds) {
  var totalMs = Math.floor(microseconds / 1000);
  var minutes = Math.floor(totalMs / 60000);
  var seconds = Math.floor((totalMs % 60000) / 1000);
  var millis = totalMs % 1000;
  var minStr = (minutes < 10 ? "0" : "") + minutes;
  var secStr = (seconds < 10 ? "0" : "") + seconds;
  var msStr = ("000" + millis).slice(-3);
  return minStr + ":" + secStr + "." + msStr;
}
while (true) {
  YIELD();
  if (msg) {
    var ts = formatTime(time);
    var module = (id == ROOT_ID) ? "Server" : "Client";
    log.log(ts + "\tID:" + id + "\t[INFO: " + module + "]\t" + msg + "\\n");
  }
}
"""
    script_js = script_js.replace("__TIMEOUT__", str(timeout_ms)).replace(
        "__ROOT__", str(server_id)
    )
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


def generate_csc_from_dict(
    topo: Dict[str, Any],
    platform_name: str,
    build_root: str | None = None,
    base_dir: str | None = None,
) -> str:
    platform = get_platform(platform_name)
    radio = topo["radio"]
    # Count clients (non-server)
    client_count = 0
    for m in topo.get("motes", []):
        if str(m.get("role", "")).lower() != "server":
            client_count += 1

    header = make_header(
        title=topo.get("topology_id", topo.get("title", "cooja_run")),
        seed=topo.get("seed", 123456),
        radio=radio,
        platform=platform,
        build_root=build_root,
        expected_nodes=client_count,
        base_dir=base_dir,
    )

    motes_xml = []
    for m in topo["motes"]:
        role = str(m.get("role", "client")).lower()
        motetype = (
            f"{platform.name}_server" if role == "server" else f"{platform.name}_client"
        )
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
    topology_json_path: str,
    out_csc_path: str,
    platform: str = "sky",
    build_root: str | None = None,
    base_dir: str | None = None,
) -> None:
    topo = json.loads(Path(topology_json_path).read_text())
    csc = generate_csc_from_dict(
        topo, platform_name=platform, build_root=build_root, base_dir=base_dir
    )
    Path(out_csc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csc_path).write_text(csc)


def main() -> None:
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
    ap.add_argument(
        "--build-root",
        type=str,
        default=None,
        help="Optional build root for per-run artifacts (e.g., <run_dir>/build)",
    )
    args = ap.parse_args()
    generate_csc(
        args.topology_json,
        args.out_csc,
        platform=args.platform,
        build_root=args.build_root,
        base_dir=args.base_dir,
    )
    print(f"Wrote {args.out_csc}")


if __name__ == "__main__":
    main()
