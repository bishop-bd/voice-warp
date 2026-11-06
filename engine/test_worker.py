"""Protocol/device policy tests; no ML packages, weights, or GPU required."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from voice_warp_backend import (
    ExplicitDeviceManager, OnnxAudit, discover_devices, exported_bundle_paths,
    select_device, validate_request, verify_acceleration,
    configure_torch_policy,
)
from worker import receive


class WorkerProtocolTests(unittest.TestCase):
    def test_rejects_truncated_oversized_and_nonobject_frames(self):
        for data in (b'{}', b'[]\n', b'x' * 300_001 + b'\n', b'invalid\n'):
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                receive(io.BytesIO(data))
        self.assertEqual(receive(io.BytesIO(b'{"protocol":2}\n')), {"protocol": 2})
        self.assertIsNone(receive(io.BytesIO()))

    def test_validates_files_and_numeric_settings_before_importing_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('voice.pth', 'hubert_base.pt', 'rmvpe.onnx'):
                (root / name).touch()
            config = dict(protocol=2, action='init', backend='cpu', device='cpu',
                          block_size=8192, pitch=0, index_ratio=0.5,
                          model_path=str(root / 'voice.pth'), assets_dir=str(root))
            validate_request(config)
            for key, value in [('block_size', True), ('pitch', True), ('pitch', 25),
                               ('index_ratio', float('nan')), ('index_ratio', True),
                               ('model_path', 'relative.pth'), ('protocol', 1),
                               ('action', 'anything'), ('backend', 'imaginary'),
                               ('device', 0), ('measure_performance', 1)]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_request({**config, key: value})
            (root / 'rmvpe.onnx').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing shared'):
                validate_request(config)

    def test_probe_requires_no_model(self):
        validate_request(dict(protocol=2, action='probe', backend='cuda', device='0'))

    def test_export_needs_cache_but_not_live_audio_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('voice.pth', 'hubert_base.pt', 'rmvpe.onnx'):
                (root / name).touch()
            config = dict(protocol=2, action='export', model_path=str(root / 'voice.pth'),
                          assets_dir=str(root), cache_dir=str(root / 'cache'))
            validate_request(config)
            for change in ({'cache_dir': 'relative'}, {'include_hubert': 1}):
                with self.assertRaises(ValueError):
                    validate_request({**config, **change})

    def test_bundle_resolves_null_paths_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {}
            for name in ('voice.onnx', 'hubert.onnx', 'rmvpe.onnx'):
                (root / name).write_bytes(name.encode())
                files[name] = hashlib.sha256(name.encode()).hexdigest()
            (root / 'export.json').write_text(json.dumps(dict(schema=1, cpu_validated=True, files=files)))
            config = dict(action='init', model_path=str(root / 'voice.onnx'), hubert_path=None)
            resolved = exported_bundle_paths(config)
            self.assertEqual(resolved['hubert_path'], str(root / 'hubert.onnx'))
            self.assertEqual(resolved['rmvpe_path'], str(root / 'rmvpe.onnx'))
            self.assertIsNone(config['hubert_path'])
            (root / 'hubert.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                exported_bundle_paths(config)

    def test_bundle_rejects_manifest_paths_outside_the_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(action='init', model_path=str(root / 'voice.onnx'))
            (root / 'voice.onnx').touch()
            for manifest in ([], dict(schema=1, cpu_validated=False),
                             dict(schema=1, cpu_validated=True,
                                  files={'voice.onnx': '', 'rmvpe.onnx': '', '../escape': ''})):
                (root / 'export.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    exported_bundle_paths(config)

    def test_worker_reports_protocol_failure_on_stdout_without_importing_ml(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py'))],
                                input=b'{"protocol":1,"action":"probe"}\n',
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        reply = json.loads(result.stdout)
        self.assertEqual(reply['protocol'], 2)
        self.assertIn('repair', reply['error'])
        self.assertIn(b'Traceback', result.stderr)

    def test_worker_exits_cleanly_on_pipe_eof(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py'))],
                                input=b'', capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')

    def test_failed_init_exits_cleanly_while_client_keeps_stdin_open(self):
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker.py'))],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            # Invalid protocol fails before importing ML, but init has already
            # started the cancellation reader, as a real initialization does.
            process.stdin.write('{"protocol":1,"action":"init"}\n')
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            self.assertIn('repair', reply['error'])
            self.assertEqual(process.wait(timeout=3), 1)
            self.assertNotIn('Fatal Python error', process.stderr.read())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_input_eof_interrupts_blocked_initialization_without_parent_death(self):
        code = (
            "import sys,time\n"
            "from worker import InputMonitor\n"
            "monitor=InputMonitor(sys.stdin.buffer)\n"
            "print('monitor ready', flush=True)\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen([sys.executable, '-c', code],
                                   cwd=Path(__file__).parent, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(process.stdout.readline(), b'monitor ready\n')
            self.assertIsNone(process.poll())
            started = time.monotonic()
            process.stdin.close()
            self.assertEqual(process.wait(timeout=3), 0)
            self.assertLess(time.monotonic() - started, 3)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_explicit_shutdown_can_be_acknowledged_after_input_closes(self):
        code = (
            "import sys,time\n"
            "from worker import InputMonitor\n"
            "monitor=InputMonitor(sys.stdin.buffer)\n"
            "request=monitor.next()\n"
            "time.sleep(0.1)\n"
            "print(request['action'],flush=True)\n"
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                input=b'{"action":"shutdown"}\n',
                                capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'shutdown\n')

    def test_completed_export_exits_while_client_keeps_stdin_open(self):
        # Isolate the lifecycle contract from expensive export dependencies.
        code = (
            "import sys\n"
            "from types import SimpleNamespace as S\n"
            "sys.modules['numpy']=S()\n"
            "sys.modules['torch']=S(set_num_threads=lambda _:None,set_num_interop_threads=lambda _:None)\n"
            "sys.modules['onnxruntime']=S()\n"
            "sys.modules['export_models']=S(export_bundle=lambda *a,**k:{'cached':True})\n"
            "import worker\n"
            "worker.protocol=sys.stdout\n"
            "worker.main()\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('voice.pth', 'hubert_base.pt', 'rmvpe.onnx'):
                (root / name).touch()
            request = dict(protocol=2, action='export', model_path=str(root / 'voice.pth'),
                           assets_dir=str(root), cache_dir=str(root / 'cache'))
            process = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            try:
                process.stdin.write(json.dumps(request) + '\n')
                process.stdin.flush()
                self.assertTrue(json.loads(process.stdout.readline())['exported'])
                self.assertEqual(process.wait(timeout=3), 0)
                self.assertNotIn('Fatal Python error', process.stderr.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.report = {'devices': [
            dict(backend='cpu', id='cpu', name='CPU', index=-1),
            dict(backend='cuda', id='uuid:a', name='Card', index=1),
            dict(backend='directml', id='luid:b', name='AMD', index=0)]}

    def test_selection_uses_identity_and_rejects_missing_gpu(self):
        self.assertEqual(select_device(self.report, 'cuda', 'uuid:a')['index'], 1)
        self.assertEqual(select_device(self.report, 'cuda', '1')['id'], 'uuid:a')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            select_device(self.report, 'cuda', '0')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            select_device(self.report, 'rocm', '0')

    def test_directml_does_not_accept_checkpoints_or_auto_choose_for_them(self):
        with self.assertRaisesRegex(ValueError, 'requires an RVC ONNX'):
            select_device(self.report, 'directml', '0', '.pth')
        only_dml = {'devices': [self.report['devices'][0], self.report['devices'][2]]}
        self.assertEqual(select_device(only_dml, 'auto', '0', '.pth')['backend'], 'cpu')
        self.assertEqual(select_device(only_dml, 'auto', '0', '.onnx')['backend'], 'directml')

    def test_ambiguous_names_do_not_silently_select_first_card(self):
        report = {'devices': [dict(backend='cuda', id='name:Card', index=i) for i in (0, 1)]}
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            select_device(report, 'cuda', 'name:Card')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            select_device(report, 'auto', 'name:Card')

    def test_rocm_uses_runtime_identity_not_cuda_namespace(self):
        torch = SimpleNamespace(
            __version__='test', version=SimpleNamespace(hip='6.2'),
            backends=SimpleNamespace(mps=None),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 get_device_name=lambda _: 'AMD',
                                 get_device_properties=lambda _: SimpleNamespace(total_memory=10, uuid='a')))
        ort = SimpleNamespace(__version__='test', get_available_providers=lambda: ['CPUExecutionProvider'])
        report = discover_devices(torch, ort)
        self.assertEqual(report['devices'][1]['backend'], 'rocm')
        self.assertFalse(report['model_verified'])

    def test_explicit_cpu_does_not_automatically_use_mps(self):
        torch = SimpleNamespace(device=lambda *args: args)
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        manager = ExplicitDeviceManager(torch, ort, self.report['devices'][0])
        self.assertEqual(manager.getDevice(-1), ('cpu',))
        self.assertFalse(manager.halfPrecisionAvailable(-1))

    def test_migraphx_pitch_policy_does_not_change_other_stage_providers(self):
        torch = SimpleNamespace(device=lambda *args: args)
        ort = SimpleNamespace(get_available_providers=lambda: ['MIGraphXExecutionProvider', 'CPUExecutionProvider'])
        manager = ExplicitDeviceManager(torch, ort, dict(backend='migraphx', index=0))
        with manager.onnx_stage('pitch'):
            self.assertEqual(manager.getOnnxExecutionProvider(0)[0], ['CPUExecutionProvider'])
            self.assertEqual(manager.device, ('cuda', 0))
        self.assertEqual(manager.getOnnxExecutionProvider(0)[0], ['MIGraphXExecutionProvider'])
        with self.assertRaises(ValueError):
            with manager.onnx_stage('pitch'):
                raise ValueError('construction failed')
        self.assertEqual(manager.getOnnxExecutionProvider(0)[0], ['MIGraphXExecutionProvider'])

    def test_acceleration_requires_a_model_stage_not_just_a_gpu_tensor(self):
        stages = {'hubert': 'CPU', 'synthesizer': 'CPU'}
        for backend in ('cuda', 'rocm', 'mps', 'directml', 'migraphx'):
            with self.subTest(backend=backend), self.assertRaisesRegex(RuntimeError, 'no HuBERT'):
                verify_acceleration(backend, stages)
        self.assertTrue(verify_acceleration('rocm', {**stages, 'hubert': 'cuda:0'}))
        self.assertTrue(verify_acceleration('directml', {**stages, 'synthesizer': 'DmlExecutionProvider + CPU nodes'}))
        self.assertFalse(verify_acceleration('cpu', stages))
        self.assertFalse(verify_acceleration('coreml', {'hubert': 'CoreMLExecutionProvider'}))

    def test_native_gpu_convolution_policy_requires_exact_qualified_target(self):
        for backend, os_name, version, architecture, applies in (
            ('rocm', 'Linux', '2.9.1+rocm7.2.1.gitff65f5bc', 'gfx1150:xnack-', True),
            ('rocm', 'Linux', '2.9.1+rocm7.2.1.gitff65f5bc', 'gfx1151', False),
            ('rocm', 'Linux', '2.9.1+rocm7.2.1.gitff65f5bc', '', False),
            ('rocm', 'Linux', '2.9.1+rocm7.2.2.gitabc', 'gfx1150', False),
            ('rocm', 'Windows', '2.9.1+rocm7.2.1.gitff65f5bc', 'gfx1150', False),
            ('cuda', 'Linux', '2.9.1+rocm7.2.1.gitff65f5bc', 'gfx1150', False),
        ):
            torch = SimpleNamespace(__version__=version,
                backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)),
                cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(gcnArchName=architecture)))
            with self.subTest(backend=backend, os=os_name, version=version, architecture=architecture), patch('voice_warp_backend.platform.system', return_value=os_name):
                policy = configure_torch_policy(torch, dict(backend=backend, index=0))
                self.assertEqual(torch.backends.cudnn.enabled, not applies)
                self.assertEqual(policy['convolution'], 'ATen GPU' if applies else 'framework default')


class AuditTests(unittest.TestCase):
    def fake_ort(self, executed=None, initialized=None):
        executed = executed or ['DmlExecutionProvider']
        created = []

        class Session:
            def __init__(self, path, sess_options, providers, provider_options, **kwargs):
                self.options, self.providers = sess_options, providers
                self.fallback_disabled = False
                created.append(self)

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_providers(self):
                return initialized or self.providers

            def end_profiling(self):
                path = self.options.profile_file_prefix + '.json'
                Path(path).write_text(json.dumps([{'args': {'provider': p}} for p in executed]))
                return path

        return SimpleNamespace(InferenceSession=Session, SessionOptions=SimpleNamespace,
                               ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
                               GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL=0)), created

    def test_directml_options_fallback_and_actual_kernel_placement(self):
        ort, created = self.fake_ort(['DmlExecutionProvider', 'CPUExecutionProvider'])
        original = ort.InferenceSession
        audit = OnnxAudit(ort)
        try:
            with audit.session_factory():
                session = ort.InferenceSession('model.onnx', providers=['DmlExecutionProvider'])
            self.assertIs(ort.InferenceSession, original)
            self.assertFalse(created[0].options.enable_mem_pattern)
            self.assertEqual(created[0].options.execution_mode, 0)
            self.assertTrue(created[0].fallback_disabled)
            self.assertEqual(audit.placement(session), 'DmlExecutionProvider + CPU nodes')
        finally:
            audit.close()

    def test_provider_list_alone_does_not_prove_gpu_execution(self):
        ort, _ = self.fake_ort(['CPUExecutionProvider'])
        audit = OnnxAudit(ort)
        try:
            with audit.session_factory():
                session = ort.InferenceSession('model.onnx', providers=['DmlExecutionProvider'])
            with self.assertRaisesRegex(RuntimeError, 'no kernels'):
                audit.placement(session)
        finally:
            audit.close()

    def test_initialization_fallback_is_rejected_and_factory_restored(self):
        ort, _ = self.fake_ort(initialized=['CPUExecutionProvider'])
        original = ort.InferenceSession
        audit = OnnxAudit(ort)
        try:
            with self.assertRaisesRegex(RuntimeError, 'silent CPU fallback'):
                with audit.session_factory():
                    ort.InferenceSession('model.onnx', providers=['DmlExecutionProvider'])
            self.assertIs(ort.InferenceSession, original)
        finally:
            audit.close()

    def test_coreml_provider_does_not_claim_physical_gpu_measurement(self):
        ort, _ = self.fake_ort(['CoreMLExecutionProvider'])
        audit = OnnxAudit(ort)
        try:
            with audit.session_factory():
                session = ort.InferenceSession('model.onnx', providers=['CoreMLExecutionProvider'])
            self.assertIn('physical GPU use not measured', audit.placement(session))
        finally:
            audit.close()

    def test_migraphx_avoids_ort_empty_shape_constant_folding(self):
        ort, created = self.fake_ort(['MIGraphXExecutionProvider'])
        audit = OnnxAudit(ort)
        try:
            with audit.session_factory():
                session = ort.InferenceSession('model.onnx', providers=['MIGraphXExecutionProvider'])
            self.assertEqual(created[0].options.graph_optimization_level, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            self.assertEqual(audit.placement(session), 'MIGraphXExecutionProvider')
        finally:
            audit.close()


if __name__ == '__main__':
    unittest.main()
