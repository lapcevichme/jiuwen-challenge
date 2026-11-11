import os
import sys
import sysconfig
import types
from pathlib import Path


def _linux(workspace: Path, scratch: Path, libraries: tuple[Path, ...], writable: bool) -> str:
    import ctypes as c
    import errno
    import platform

    if platform.machine() != "x86_64":
        return "sandbox off: not x86-64"

    libc = c.CDLL(None, use_errno=True)
    libc.syscall.restype = c.c_long

    def checked(result):
        if result < 0:
            error = c.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    status = []
    checked(libc.prctl(38, 1, 0, 0, 0))
    try:
        status.append(_landlock(c, checked, libc, workspace, scratch, libraries, writable))
    except OSError as error:
        status.append(f"landlock off ({error.strerror})")
    try:
        _seccomp(c, checked, libc, errno)
        status.append("seccomp on")
    except OSError as error:
        status.append(f"seccomp off ({error.strerror})")
    return ", ".join(status)


def _landlock(c, checked, libc, workspace, scratch, libraries, writable) -> str:
    abi = checked(libc.syscall(444, 0, 0, 1))
    handled_bits = {1: 13, 2: 14}.get(abi, 15)

    class PathRule(c.Structure):
        _fields_ = [("access", c.c_uint64), ("parent_fd", c.c_int32)]

    execute, read_file, read_dir = 1 << 0, 1 << 2, 1 << 3
    read = read_file | read_dir
    write = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 13) | (1 << 14)
    mask = (1 << handled_bits) - 1
    handled = c.c_uint64(mask)
    system = [Path(path) for path in ("/usr", "/lib", "/lib64", "/bin", "/sbin") if os.path.isdir(path)]
    system.append(Path(sys.executable).resolve().parent)
    venv = Path(sys.executable).parent.parent / "pyvenv.cfg"
    files = [path for path in (Path("/etc/ld.so.cache"), venv) if path.is_file()]
    rules = checked(libc.syscall(444, c.byref(handled), c.sizeof(handled), 0))
    try:
        for path, access in [(workspace, read | (write if writable else 0)), (scratch, read | write),
                             *((path, read) for path in libraries), *((path, read | execute) for path in system),
                             *((path, read_file) for path in files)]:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access & mask, fd)
                checked(libc.syscall(445, rules, 1, c.byref(rule), 0))
            finally:
                os.close(fd)
        checked(libc.syscall(446, rules, 0))
    finally:
        os.close(rules)
    return f"landlock abi {abi}"


def _seccomp(c, checked, libc, errno) -> None:
    class Filter(c.Structure):
        _fields_ = [("code", c.c_ushort), ("jt", c.c_uint8), ("jf", c.c_uint8), ("k", c.c_uint32)]

    class Program(c.Structure):
        _fields_ = [("length", c.c_ushort), ("filters", c.POINTER(Filter))]

    deny = (
        *range(41, 56),
        62, 101, 129, 200, 234, 297, 310, 311,
        90, 91, 92, 93, 94, 132, *range(188, 191), *range(197, 200), 235, 260, 261, 268, 280,
        452, 463, 466, 469,
        155, 161, 165, 166, 272, 303, 304, 308, 323, 442, 467, 472,
        288, 299, 307,
        424, *range(425, 435), 438,
    )
    blocked = 0x00050000 | errno.EPERM
    allowed = 0x7FFF0000
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, blocked),
                    (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, blocked)]
    instructions.extend(((0x15, 0, 4, 16), (0x20, 0, 0, 24), (0x15, 0, 1, 0x5451),
                         (0x06, 0, 0, allowed), (0x06, 0, 0, blocked)))
    for number in deny:
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, blocked)))
    instructions.append((0x06, 0, 0, allowed))
    filters = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
    checked(libc.prctl(22, 2, c.byref(Program(len(filters), filters)), 0, 0))


def _windows_guard(workspace: Path, scratch: Path, libraries: tuple[Path, ...], writable: bool) -> None:
    write_roots = (workspace, scratch) if writable else (scratch,)
    read_roots = (workspace, scratch, *libraries)
    blocked_modules = {"ctypes", "_ctypes", "_winapi", "msvcrt", "winreg", "mmap", "_socket", "_overlapped"}
    mutations = {"os.chmod", "os.mkdir", "os.remove", "os.rmdir", "os.utime", "os.truncate"}

    def check(path, roots):
        if isinstance(path, int):
            raise PermissionError("task sandbox: direct file descriptor access denied")
        resolved = Path.cwd() if path is None else Path(os.fsdecode(path)).resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            raise PermissionError("task sandbox: path outside allowed directories")

    def audit(event, args):
        if event == "open":
            flags = args[2]
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            check(args[0], write_roots if writing else read_roots)
        elif event == "import":
            if args[0].split(".")[0] in blocked_modules:
                raise ImportError("task sandbox: native access denied")
            if isinstance(args[1], str) and args[1].lower().endswith(".pyd"):
                check(args[1], libraries)
        elif event in {"os.listdir", "os.scandir"}:
            check(args[0], read_roots)
        elif event == "os.chdir":
            check(args[0], (workspace, scratch))
        elif event in mutations:
            check(args[0], write_roots)
        elif event == "os.rename":
            for path in args[:2]:
                check(path, write_roots)
        elif event.startswith(("socket.", "subprocess.", "ctypes.", "winreg.")) or event in {
            "os.system", "os.startfile", "os.startfile/2", "os.exec", "os.spawn", "os.kill",
            "os.link", "os.symlink", "os.add_dll_directory",
        }:
            raise PermissionError("task sandbox: external operation denied")

    sys.addaudithook(audit)


def main() -> None:
    script, scratch, writable = sys.argv[1:]
    scratch = Path(scratch).resolve()
    workspace = Path.cwd().resolve()
    probe = script == "--probe"
    script = Path(script)
    code = None if probe else compile(script.read_bytes(), str(script), "exec")
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    libraries = (stdlib,)
    if sys.platform == "win32":
        libraries += (Path(sys.base_prefix, "DLLs").resolve(),)

    program = types.ModuleType("__main__")
    program.__file__ = str(script)
    sys.argv[:] = [str(script)]
    sys.path.insert(0, str(script.parent if script.is_relative_to(workspace) else workspace))
    if os.environ.get("HARNESS_SANDBOX") == "0":
        status = "sandbox off: HARNESS_SANDBOX=0"
    else:
        try:
            if sys.platform == "linux":
                status = _linux(workspace, scratch, libraries, writable == "1")
            elif sys.platform == "win32":
                _windows_guard(workspace, scratch, libraries, writable == "1")
                status = "windows audit hook"
            else:
                status = f"sandbox off: unsupported platform {sys.platform}"
        except (OSError, RuntimeError) as error:
            status = f"sandbox off: {error}"
    if probe:
        print(status)
        return
    sys.modules["__main__"] = program
    exec(code, program.__dict__)


if __name__ == "__main__":
    main()
