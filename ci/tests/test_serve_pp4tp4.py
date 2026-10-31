"""Host-only checks for ci/serve/pp4tp4; no Docker, network, or NPU use."""

import subprocess
import tempfile
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
SERVE = CI / "serve"
PP = SERVE / "pp4tp4"
DP = SERVE / "dp8tp4ep32"


def run(*args):
    return subprocess.run(args, capture_output=True, text=True)


class LayoutTests(unittest.TestCase):
    def test_directory_mirrors_the_decode_launcher(self):
        # Same shape as dp8tp4ep32 so operators move between the two topologies
        # without relearning the layout.
        for name in ("README.md", "container/serve.sh", "container/cluster.env.example",
                     "container/speculative_config.synthetic.json",
                     "container/speculative_config.standard.json"):
            self.assertTrue((DP / name).is_file(), name)
            self.assertTrue((PP / name).is_file(), name)
        self.assertTrue((PP / "start-pp4tp4.sh").is_file())
        self.assertTrue((PP / "stop-pp4tp4.sh").is_file())
        self.assertTrue((PP / "prefill_bench.py").is_file())
        self.assertTrue((PP / "start-pp4tp4.sh").stat().st_mode & 0o100)
        self.assertTrue((PP / "stop-pp4tp4.sh").stat().st_mode & 0o100)

    def test_shell_syntax(self):
        for path in (PP / "start-pp4tp4.sh", PP / "stop-pp4tp4.sh",
                     PP / "container/serve.sh"):
            result = run("bash", "-n", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prefill_client_parses(self):
        # Parse in-process: py_compile would write __pycache__ into the repo,
        # which fails when the checkout is root-owned.
        source = (PP / "prefill_bench.py").read_text()
        result = subprocess.run(
            ["python3", "-c", "import ast,sys; ast.parse(sys.stdin.read())"],
            input=source, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TopologyTests(unittest.TestCase):
    def test_cluster_env_declares_pp4tp4_on_two_nodes(self):
        env = (PP / "container/cluster.env.example").read_text()
        for line in ("NODE_IPS=(10.208.128.40 10.208.128.41)",
                     "PP_SIZE=4", "TP_SIZE=4", "DP_SIZE=1",
                     'PP_LAYER_PARTITION="24,25,25,18"',
                     "MAX_BATCHED_TOKENS=16384", "API_PORT=10086", "PP_MASTER_PORT=7060",
                     "MAX_NUM_SEQS=64"):
            self.assertIn(line, env)

    def test_layer_partition_covers_the_decoder_stack(self):
        env = (PP / "container/cluster.env.example").read_text()
        line = next(l for l in env.splitlines() if l.startswith("PP_LAYER_PARTITION="))
        parts = [int(x) for x in line.split('"')[1].split(",")]
        self.assertEqual(len(parts), 4, "one entry per pipeline stage")
        self.assertEqual(sum(parts), 92, "must partition the 92 decoder layers")

    def test_force_eplb_is_a_single_switch(self):
        # eplb is compared, not assumed: one value drives additional-config so a
        # comparison run differs in exactly that setting.
        serve = (PP / 'container/serve.sh').read_text()
        self.assertIn('FORCE_EPLB', serve)
        # The JSON lives inside a shell string, so each quote keeps a backslash
        # in the file; match the value interpolation, not the escaped form.
        self.assertIn('enable_force_eplb\\":$EPLB_JSON', serve)
        # The hardcoded literal must be gone or the switch cannot take effect.
        self.assertNotIn('enable_force_eplb\\":false}', serve)
        self.assertIn('--additional-config "$ADDITIONAL_CONFIG_JSON"', serve)
        start = (PP / 'start-pp4tp4.sh').read_text()
        self.assertIn('--force-eplb) FORCE_EPLB=1; shift ;;', start)
        self.assertIn('FORCE_EPLB', start.split('overrides=')[1].split('"')[1])
        self.assertIn('echo "FORCE_EPLB=$FORCE_EPLB"', start)
        self.assertIn('FORCE_EPLB must be 0 or 1', start)
        env = (PP / 'container/cluster.env.example').read_text()
        self.assertIn('FORCE_EPLB=0', env)

    def test_prefill_launcher_keeps_the_plugin_validated_settings(self):
        # These three were each observed to break PP4 startup; keep them pinned
        # so a later edit cannot quietly reintroduce them.
        serve = (PP / 'container/serve.sh').read_text()
        # PP>1 + MTP requires the target readback; 0 raises NotImplementedError.
        self.assertIn('VLLM_ASCEND_TARGET_EXACT_SEQ_LENS=1', serve)
        # The prefill slot allocator key is not in the plugin's config schema
        # and fails pydantic validation on node1.
        self.assertNotIn('step5_prefill_mnd_slot_allocator', serve)
        # bs=64 needs a scheduler queue deep enough not to serialise the tail.
        env = (PP / 'container/cluster.env.example').read_text()
        self.assertIn('MAX_NUM_SEQS=64', env)

    def test_health_probe_runs_on_node0(self):
        # The API port lives on the cluster network; probing from the calling
        # host hangs until the timeout even after the service is serving.
        start = (PP / 'start-pp4tp4.sh').read_text()
        self.assertIn('probe_health()', start)
        # Slice on a line boundary: the body contains ${...}, so splitting on
        # the first "}" would cut inside the expansion.
        probe = start.split('probe_health() {', 1)[1].split('\n}', 1)[0]
        self.assertIn('remote_sudo', probe)
        # Must target the bound address: the service binds --host "$LOCAL_IP",
        # so 127.0.0.1 refuses the connection.
        self.assertIn('http://${api_ip}:${api_port}/health', probe)
        self.assertNotIn('127.0.0.1:', probe)
        self.assertNotIn('curl -s -m 5 -o /dev/null -w \'%{http_code}\' "http://${api_ip}', start)

    def test_serve_sh_launches_a_two_node_pipeline(self):
        serve = (PP / "container/serve.sh").read_text()
        for flag in ("--pipeline-parallel-size", "--nnodes 2", "--node-rank",
                     "--master-addr", "--master-port", "--headless"):
            self.assertIn(flag, serve)
        self.assertIn('[[ "$PP_SIZE" == 4 && "$TP_SIZE" == 4 && "$DP_SIZE" == 1', serve)
        self.assertIn('${#NODE_IPS[@]}', serve)
        self.assertIn("VLLM_PP_LAYER_PARTITION", serve)
        # PP>1 + MTP rejects the decode-only 0 readback setting.
        self.assertIn("VLLM_ASCEND_TARGET_EXACT_SEQ_LENS=1", serve)
        # One DP replica: the size is passed explicitly (as in pp1.sh) but the
        # multi-node DP wiring belongs to the decode topology only.
        self.assertIn('--data-parallel-size "$DP_SIZE"', serve)
        self.assertNotIn("--data-parallel-address", serve)
        self.assertNotIn("--data-parallel-rpc-port", serve)
        self.assertNotIn("--data-parallel-size-local", serve)
        self.assertNotIn("--data-parallel-backend", serve)

    def test_launcher_hands_out_the_same_topology_to_both_nodes(self):
        start = (PP / "start-pp4tp4.sh").read_text()
        self.assertIn("for idx in 1 0; do", start, "node0 must start last")
        self.assertIn("Image digest differs across nodes", start)
        self.assertIn("--run-dir PATH", start)
        # serve.sh indexes LOCAL_IP by NODE_ID; the container also needs the
        # weights mounted and a writable JIT cache, or it starts with no model.
        self.assertIn("-e LOCAL_IP=$ip", start)
        self.assertIn('-v "$model_host":/model/target:ro', start)
        self.assertIn("TORCH_EXTENSIONS_DIR=/run-cluster/torch-extensions", start)
        self.assertIn("MODEL_PATH_HOST missing from cluster.env", start)
        # The launcher may run on either node; half of it must not need root
        # self-SSH (which a fresh node does not have).
        self.assertIn("is_local()", start)
        self.assertIn("node_sh", start)
        self.assertIn("remote_sudo()", start)
        self.assertEqual(start.count("--ips needs exactly two addresses"), 1)
        stop = (PP / "stop-pp4tp4.sh").read_text()
        self.assertIn("for idx in 0 1; do", stop, "node0 must stop first")


class CliTests(unittest.TestCase):
    def test_dry_run_prints_the_plan(self):
        for extra in ([], ["--image", "example.invalid/step5:test"]):
            result = run("bash", str(PP / "start-pp4tp4.sh"),
                         "--ips", "10.0.0.1,10.0.0.2", "--dry-run", *extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry run: nothing was started.", result.stdout)
            self.assertIn("10.0.0.1 10.0.0.2", result.stdout)

    def test_help(self):
        for script in ("start-pp4tp4.sh", "stop-pp4tp4.sh"):
            result = run("bash", str(PP / script), "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--ips", result.stdout)

    def test_rejects_bad_invocations(self):
        cases = [
            [],
            ["--ips", "10.0.0.1"],
            ["--ips", "10.0.0.1,10.0.0.2,10.0.0.3"],
            ["--ips", "10.0.0.1,10.0.0.1"],
            ["--ips", "10.0.0.1,10.0.0.300"],
            ["--ips", "10.0.0.1,10.0.0.2", "--unknown"],
            ["--ips", "10.0.0.1,10.0.0.2", "--spec-config", "bogus"],
            ["--ips", "10.0.0.1,10.0.0.2", "--spec-config", "/nonexistent/spec.json"],
        ]
        for argv in cases:
            result = run("bash", str(PP / "start-pp4tp4.sh"), *argv, "--dry-run")
            self.assertNotEqual(result.returncode, 0, argv)

    def test_spec_config_validation_rejects_broken_files(self):
        with tempfile.TemporaryDirectory() as temp:
            broken = Path(temp) / "spec.json"
            broken.write_text('{"method": "eagle", "rejection_sample_method": "standard"}')
            result = run("bash", str(PP / "start-pp4tp4.sh"),
                         "--ips", "10.0.0.1,10.0.0.2",
                         "--spec-config", str(broken), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid MTP config", result.stderr)


if __name__ == "__main__":
    unittest.main()
