"""Public LoopAct model components, loaded on demand."""
import loopact as _loopact
__all__ = _loopact.__all__

def __getattr__(name):
    return getattr(_loopact, name)
