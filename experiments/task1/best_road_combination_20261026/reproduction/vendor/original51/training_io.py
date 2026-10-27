"""OS-owned file locks: automatically released when the process exits."""
import os
from pathlib import Path

class RunLock:
    def __init__(self,path): self.path=Path(path); self.handle=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b')
        self.handle.seek(0,2)
        if self.handle.tell()==0:
            self.handle.write(b'0'); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close(); self.handle=None
            raise RuntimeError(f'Another process is using this output directory: {self.path.parent}') from error
        return self
    def release(self):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
            self.handle.close(); self.handle=None
    def __enter__(self): return self.acquire()
    def __exit__(self,*args): self.release()
