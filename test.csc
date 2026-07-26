<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>star_n10_s10_seed123456</title>
    <randomseed>123456</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>100.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>

    <motetype>
      org.contikios.cooja.mspmote.SkyMoteType
      <identifier>sky_server</identifier>
      <description>RPL Server (Root) - SKY</description>
      <source>/home/duy/RPL-TinyML/rpl/motes/udp-server.c</source>
      <commands>$(MAKE) -C /home/duy/RPL-TinyML/rpl/motes -j$(CPUS)  DEFINES+=EXPECTED_NODES=9 udp-server.sky TARGET=sky</commands>
      <firmware>/home/duy/RPL-TinyML/rpl/motes/build/sky/udp-server.sky</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspClock</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyButton</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyFlash</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.Msp802154Radio</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDefaultSerial</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspLED</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDebugOutput</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyLED</moteinterface>
    </motetype>

    <motetype>
      org.contikios.cooja.mspmote.SkyMoteType
      <identifier>sky_client</identifier>
      <description>RPL Client - SKY</description>
      <source>/home/duy/RPL-TinyML/rpl/motes/udp-server.c</source>
      <commands>$(MAKE) -C /home/duy/RPL-TinyML/rpl/motes -j$(CPUS) udp-server.sky TARGET=sky</commands>
      <firmware>/home/duy/RPL-TinyML/rpl/motes/build/sky/udp-server.sky</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspClock</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyButton</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyFlash</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.Msp802154Radio</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDefaultSerial</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspLED</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDebugOutput</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.SkyLED</moteinterface>
    </motetype>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>0.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>1</id>
    </interface_config>
    <motetype_identifier>sky_server</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>10.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>2</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>7.66044443118978</x>
      <y>6.4278760968653925</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>3</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>1.7364817766693041</x>
      <y>9.84807753012208</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>4</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>-4.999999999999998</x>
      <y>8.660254037844387</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>5</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>-9.396926207859083</x>
      <y>3.420201433256689</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>6</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>-9.396926207859085</x>
      <y>-3.4202014332566866</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>7</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>-5.000000000000004</x>
      <y>-8.660254037844384</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>8</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>1.7364817766692997</x>
      <y>-9.848077530122081</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>9</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>7.660444431189778</x>
      <y>-6.427876096865396</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>10</id>
    </interface_config>
    <motetype_identifier>sky_client</motetype_identifier>
  </mote>
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script><![CDATA[

// Headless logging for parser compatibility
TIMEOUT(180000, log.testOK());
var ROOT_ID = 1;
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
    log.log(ts + "	ID:" + id + "	[INFO: " + module + "]	" + msg + "\n");
  }
}

      ]]></script>
      <active>true</active>
    </plugin_config>
    <bounds x="0" y="0" height="100" width="100" />
  </plugin>
</simconf>