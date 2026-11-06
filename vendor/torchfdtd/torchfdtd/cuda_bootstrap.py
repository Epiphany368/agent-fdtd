"""Initialize optional CUDA imports without retaining a workload call stack.

Some optional-dependency import failures are retained by CuPy as exception
objects. Their traceback frames can retain the entire importing caller stack.
Import on a dedicated zero-argument thread before constructing solver buffers.
The worker imports a module only. It does not select a device, create a CUDA
context explicitly, launch kernels, or allocate field tensors.
"""
from importlib import import_module
import threading

_lock = threading.Lock()
_module = None
_failure = None


def _import_worker():
    """Module-level zero-argument target, never a closure over caller inputs."""
    global _module, _failure
    try:
        _module = import_module('cupy')
    except BaseException as error:
        # Never store the exception or its traceback in our module/thread state.
        # In particular, do not return it as an exception cause to the caller.
        try:
            detail = str(error)
        except BaseException:
            detail = 'Exception message unavailable.'
        _failure = f'{type(error).__name__}: {detail}'


def prepare_cuda_kernels():
    """Return the imported CuPy module, initializing it once across callers.

    Call at runtime, before solver buffer construction, rather than from a
    module import that holds an import lock needed by CuPy. Successful calls
    are cached. Failed imports may be retried after fixing the environment.
    Errors preserve their type/message as text, without retaining a worker
    exception. This cannot undo frames retained by an earlier external CuPy
    import. No private dependency state is modified.
    """
    global _failure
    with _lock:
        if _module is not None:
            return _module
        _failure = None
        worker = threading.Thread(target=_import_worker, name='torchfdtd-cuda-import')
        worker.start()
        worker.join()
        if _module is None:
            detail = _failure or 'The import worker exited without a module.'
            raise RuntimeError(
                'CUDA kernel dependency initialization failed. Install '
                'torchfdtd[cuda-kernels] with a CuPy build compatible with your '
                f'CUDA environment, then retry. Import reported {detail}'
            ) from None
        return _module
