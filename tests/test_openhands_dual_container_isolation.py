import inspect

from featurebench.infer import run_infer
from featurebench.infer.agents import openhands
from featurebench.infer.container import ContainerManager


def test_embedded_runner_uses_sandbox_proxy_tools():
    script = openhands._OPENHANDS_SDK_RUNNER_SCRIPT

    compile(script, "openhands-sdk-runner.py", "exec")
    assert "get_default_tools" not in script
    assert "class SandboxTerminalExecutor" in script
    assert "class SandboxFileEditorExecutor" in script
    assert 'register_tool("terminal", SandboxTerminalTool)' in script
    assert 'register_tool("file_editor", SandboxFileEditorTool)' in script
    # The controller may contain site-packages for OpenHands dependencies, so
    # exposing the SDK's default local terminal/editor would reintroduce leakage.
    assert "Tool(name=\"terminal\")" in script
    assert "Tool(name=\"file_editor\")" in script


def test_embedded_runner_blocks_controller_leak_paths():
    script = openhands._OPENHANDS_SDK_RUNNER_SCRIPT

    assert "/opt/openhands-venv" in script
    assert "/var/run/docker.sock" in script
    assert "/installed-agent" in script
    assert "/agent-logs" in script
    assert "Path {raw} escapes the /testbed sandbox." in script
    assert "All paths must be absolute and resolve under /testbed." in script


def test_openhands_env_setup_exports_sandbox_identity():
    agent = openhands.OpenHandsAgent(
        container_manager=None,
        env_vars={
            "LLM_API_KEY": "key",
            "SANDBOX_CONTAINER_ID": "sandbox-123",
            "SANDBOX_WORKSPACE": "/testbed",
        },
        logger=None,
        model="openai/test",
    )

    env_script = agent.get_env_setup_script()

    assert "export SANDBOX_CONTAINER_ID='sandbox-123'" in env_script
    assert "export SANDBOX_WORKSPACE='/testbed'" in env_script


def test_container_manager_accepts_explicit_network_mode(monkeypatch):
    captured = {}

    class FakeImages:
        def get(self, image_name):
            return object()

    class FakeContainers:
        def run(self, **kwargs):
            captured.update(kwargs)

            class FakeContainer:
                short_id = "abc123"

            return FakeContainer()

    class FakeClient:
        images = FakeImages()
        containers = FakeContainers()

        def version(self):
            return {"ApiVersion": "1.41"}

    monkeypatch.setattr("featurebench.infer.container.docker.from_env", lambda: FakeClient())

    manager = ContainerManager(env_vars={})
    manager.create_container("image:latest", network_mode="none")

    assert captured["network_mode"] == "none"


def test_container_manager_disconnects_existing_networks(monkeypatch):
    disconnected = []

    class FakeNetwork:
        def __init__(self, name):
            self.name = name

        def disconnect(self, container, force=False):
            disconnected.append((self.name, container.short_id, force))

    class FakeNetworks:
        def get(self, name):
            return FakeNetwork(name)

    class FakeClient:
        networks = FakeNetworks()

    class FakeContainer:
        short_id = "sandbox-1"
        attrs = {"NetworkSettings": {"Networks": {"bridge": {}, "fb-net": {}}}}

        def reload(self):
            return None

    monkeypatch.setattr("featurebench.infer.container.docker.from_env", lambda: FakeClient())

    manager = ContainerManager(env_vars={})
    manager.disconnect_container_networks(FakeContainer())

    assert disconnected == [
        ("bridge", "sandbox-1", True),
        ("fb-net", "sandbox-1", True),
    ]


def test_openhands_process_single_task_uses_sandbox_for_runtime_completion():
    source = inspect.getsource(run_infer.InferenceRunner._process_single_task)

    assert "use_dual_container_openhands = self.config.agent == \"openhands\"" in source
    assert "agent.run_with_sandbox(" in source
    assert "runtime_handler.complete_runtime(sandbox_container, instance, log_file)" in source
    assert "\"SANDBOX_CONTAINER_ID\": sandbox_container.id" in source
    assert "if not use_dual_container_openhands" in source
    assert "controller_volumes" in source
    assert "\"bind\": \"/download\"" in source
    assert "cm.disconnect_container_networks(sandbox_container)" in source


def test_openhands_run_with_sandbox_collects_controller_artifacts():
    source = inspect.getsource(openhands.OpenHandsAgent.run_with_sandbox)

    assert "self.prepare_run(controller_container, instruction, log_file)" in source
    assert "self.post_run_hook(controller_container, log_file)" in source
    assert "cd /agent-logs/workspace" in source
