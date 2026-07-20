TIMEOUT(300000); /* Time in milliseconds (e.g., 60 seconds) */

while (true) {
  log.log(time + ":" + id + ":" + msg + "\n");
  YIELD(); /* Wait for next mote output */
  if (time > 120000000) { 
    log.log("Simulation finished successfully.\n");
    log.testOK(); /* CRITICAL: Tells Cooja the test passed */
  }
}
