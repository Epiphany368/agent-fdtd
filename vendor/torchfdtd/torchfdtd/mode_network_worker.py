"""Owned spawned process for interruptible modal solves, with JSON-only IPC."""
import json
from pathlib import Path


MAX_MESSAGE_BYTES = 65536


def send_message(connection, message):
    payload = json.dumps(message, allow_nan=False, separators=(',', ':')).encode('utf8')
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError('Modal worker progress exceeds the IPC message limit.')
    connection.send_bytes(payload)


def mode_network_worker(snapshot, output_path, connection, memory_admission=False):
    """Only the parent-selected generated scratch path is written by this child."""
    try:
        from .models import server_limits
        from .mode_network_project import ModeNetworkConfig, run_mode_network
        # Only the workbench server spawns this worker; its request limits (or its memory admission) apply here too.
        with server_limits(memory_admission=memory_admission):
            config = ModeNetworkConfig.model_validate(snapshot)
            result = run_mode_network(config, on_progress=lambda data:
                send_message(connection, {'type': 'progress', 'data': data}))
        if result.get('status') != 'completed':
            raise RuntimeError('Modal adapter did not complete the requested calculation.')
        payload = json.dumps(result, allow_nan=False).encode('utf8')
        if len(payload) > config.execution.output_budget_bytes:
            raise ValueError('Modal JSON result exceeds the configured output budget.')
        Path(output_path).write_bytes(payload)
        send_message(connection, {'type': 'completed'})
    except BaseException as exc:
        try:
            send_message(connection, {'type': 'error',
                'error': f'{type(exc).__name__}: {str(exc)[:4000]}'})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def close_owned_process(process):
    """Reap this process only, including cancellation inside native solver code."""
    if process is None:
        return
    if process.pid is not None:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError('Owned modal worker did not exit after cancellation.')
    process.close()
