"""Default stdout callback, minus the "FAILED - RETRYING" spam.

simulate_seeds.yml polls simulate.py's async job with an `until`/`retries`
loop (every 30s, for up to 24h). ansible-core's stock `default` callback
prints a "FAILED - RETRYING: ..." line on every single poll unconditionally
-- there is no ini/env setting or per-task option (no_log included) that
suppresses it, since v2_runner_retry() in
ansible/plugins/callback/default.py calls self._display.display() with no
verbosity or no_log gate. Subclassing it and no-opping that one hook is the
only way to keep everything else about the default output.
"""

# Options (display_skipped_hosts, show_per_host_start, etc.) are read from
# *this* plugin's own DOCUMENTATION, not inherited from the parent class --
# without it, get_option() KeyErrors on every option `default` declares via
# the default_callback doc fragment.
DOCUMENTATION = """
    name: quiet_retries
    type: stdout
    short_description: default Ansible screen output, minus retry spam
    version_added: historical
    description:
        - Same as the built-in C(default) stdout callback, except it drops
          the C(FAILED - RETRYING) line that C(until)/C(retries) loops print
          on every poll.
    extends_documentation_fragment:
      - default_callback
      - result_format_callback
    requirements:
      - set as stdout in configuration
"""

from ansible.plugins.callback.default import CallbackModule as Default


class CallbackModule(Default):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = "stdout"
    CALLBACK_NAME = "quiet_retries"

    def v2_runner_retry(self, result):
        pass
