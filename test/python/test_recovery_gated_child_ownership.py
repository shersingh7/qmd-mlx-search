"""Real-kernel regressions for interrupted gated-child acquisition.

The live-miss injector deliberately omits one owned child from ONE membership
snapshot. It does not fake waitid/waitpid, protocol messages or cleanup proof.
The original native snapshot's reason is not asserted by this regression.
This file is also the supervised fixture used by the cross-runtime API test.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

SCENARIOS = ("no-child", "live-control", "exiting-miss", "live-miss")


def supervised_fixture(source: Path, scenario: str, root: Path, proof_fd: int, config: str) -> None:
    assert scenario in SCENARIOS
    if proof_fd != 3:
        os.dup2(proof_fd, 3)
        os.close(proof_fd)
    sys.path.insert(0, str(source.parent))
    scope = {"__name__": "gated_child_regression"}
    code = source.read_bytes()
    exec(compile(code, str(source), "exec"), scope)
    sys.argv = [str(source), config]
    log_path = root / "kernel.jsonl"

    def record(kind, **fields):
        with log_path.open("a") as log:
            log.write(json.dumps(dict(kind=kind, **fields)) + "\n")

    record("runtime", executable=sys.executable, version=sys.version, platform=sys.platform,
           source=str(source), sourceSha256=hashlib.sha256(code).hexdigest(), scenario=scenario)
    real_fork, real_waitpid, real_kill = os.fork, os.waitpid, os.kill
    real_members = scope["session_groups"]
    real_observe = scope["wait_child_without_reaping"]
    release_read, release_write = os.pipe()
    children = []
    omitted = False
    released = False

    def release():
        nonlocal released
        if not released:
            os.close(release_write)
            released = True
            record("holder-released")

    def fork():
        if children:  # Do not apply the worker injector to joining anchors.
            pid = real_fork()
            if pid:
                children.append(pid)
            return pid
        if scenario == "no-child":
            record("boundary", stage="before-kernel-fork", holdSeconds=1.300)
            time.sleep(1.300)  # Real caller cancellation must interrupt this.
        pid = real_fork()
        if not pid:
            os.close(release_write)
            if scenario == "exiting-miss":
                os._exit(125)  # An actual exit, not an authored wait status.
            os.read(release_read, 1)  # Owned holder survives until first wait proof.
            os.close(release_read)
            return 0
        children.append(pid)
        os.close(release_read)
        if scenario == "exiting-miss":
            limit = time.monotonic() + 0.100
            while real_observe(pid) is None and time.monotonic() < limit:
                time.sleep(0.001)
            assert real_observe(pid) == 125, "Exiting-child injector missed its kernel boundary"
            record("owned-exit-before-cancel", pid=pid, exitCode=125, unreaped=True)
        record("boundary", stage="fork-return", pid=pid, holdSeconds=1.300)
        time.sleep(1.300)  # Interrupted before os.fork() returns to STORE_FAST.
        return pid

    def members(session_id):
        nonlocal omitted
        found = real_members(session_id)
        if children and not omitted and scenario in ("live-miss", "exiting-miss"):
            pid = children[0]
            status = real_observe(pid)  # Non-consuming REAL kernel authentication.
            # Darwin may make an unreaped zombie invisible to getsid/PGID
            # inspection. Only live-child controls require live membership;
            # the exiting control's authority is its actual WNOWAIT exit.
            same_session = os.getsid(pid) == session_id if status is None else None
            assert same_session is not False
            assert (status is None) == (scenario == "live-miss")
            record("discovery-omission", pid=pid, actualListed=pid in found,
                   sameSession=same_session, ownedUnreaped=True, observedExit=status,
                   injection="omit one exact owned child from first snapshot")
            found.pop(pid, None)
            omitted = True
        return found

    def observe(pid):
        status = real_observe(pid)
        record("waitid", pid=pid, exitCode=status, nonConsuming=True)
        if children and pid == children[0] and scenario == "live-control":
            release()
        return status

    def waitpid(pid, flags):
        result = real_waitpid(pid, flags)
        record("waitpid", requestedPid=pid, flags=flags, returnedPid=result[0], status=result[1])
        if pid == -1 and flags == os.WNOHANG and scenario == "live-miss" and not released:
            assert result[0] == 0, "Live-child injector missed WNOHANG=0"
            assert real_observe(children[0]) is None, "Holder exited before ownership observation"
            record("owned-live-at-zero", pid=children[0], ownedUnreaped=True)
            release()  # Product gate is already closed; this never grants E.
        return result

    def kill(pid, number):
        # This is evidence of product delivery, not fixture cleanup.
        record("product-direct-signal", pid=pid, signal=int(number),
               ownedUnreaped=real_observe(pid) is not None or pid in children)
        return real_kill(pid, number)

    os.fork, os.waitpid, os.kill = fork, waitpid, kill
    scope["session_groups"], scope["wait_child_without_reaping"] = members, observe
    try:
        scope["main"]()
        record("product-return")
    except BaseException as failure:
        record("product-failure", type=type(failure).__name__, error=str(failure))
        raise
    finally:
        # No cleanup signal is allowed after wait ownership was consumed. This
        # retained owner, not a later PID listing, also makes RED assertion-safe.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.fork, os.waitpid, os.kill = real_fork, real_waitpid, real_kill
        release()
        recovered = []
        for pid in children:
            try:
                status = real_observe(pid)
            except OSError as error:
                if error.errno != errno.ECHILD:
                    raise
                record("fixture-already-reaped", pid=pid, kernelECHILD=True)
                continue
            signalled = status is None
            if signalled:
                real_kill(pid, signal.SIGKILL)  # Exact retained direct child.
            actual_pid, status = real_waitpid(pid, 0)
            recovered.append(dict(pid=actual_pid, status=status, signalled=signalled))
        if not children:
            os.close(release_read)
        record("fixture-cleanup", recovered=recovered)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--supervisor":
    supervised_fixture(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    raise SystemExit(0)


# The supervisor does not need the test driver before its caller's deadline.
import select
import subprocess
import tempfile
import traceback
import unittest


def run_case(source: Path, scenario: str, root: Path) -> dict:
    read_fd = write_fd = None
    child = None
    streams = ()
    # First acquisition/setup belongs to the same scope as body and teardown.
    try:
        root.mkdir(parents=True, exist_ok=False)
        read_fd, write_fd = os.pipe()
        command = ["/usr/bin/python3", "-S", "-u", str(Path(__file__).resolve()), "--supervisor",
                   str(source), scenario, str(root), str(write_fd), json.dumps(dict(
                       command=["/usr/bin/python3", "-S", "-c", "raise RuntimeError('Gated work escaped')"],
                       timeoutMs=200, contained=False))]
        started = time.monotonic()
        # The invoking harness supplies a synthetic/offline allowlisted environment.
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True,
                                 pass_fds=(write_fd,), env=dict(os.environ))
        streams = (child.stdin, child.stdout, child.stderr)
        owned_write, write_fd = write_fd, None
        os.close(owned_write)  # Never retry an ambiguous close on a reused number.
        os.set_blocking(read_fd, False)
        proof, buffered = [], b""
        cancelled = False
        guard_fired = False
        while child.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= 0.200 and not cancelled:
                cancelled = True
                child.stdin.write(b"T")
                child.stdin.flush()
                child.send_signal(signal.SIGTERM)
                child.send_signal(signal.SIGCONT)
            if elapsed >= 1.800:
                guard_fired = True
                child.send_signal(signal.SIGTERM)
                child.send_signal(signal.SIGCONT)
                break
            ready, _, _ = select.select([read_fd], [], [], 0.001)
            if ready:
                chunk = os.read(read_fd, 65536)
                buffered += chunk
                while b"\n" in buffered:
                    line, buffered = buffered.split(b"\n", 1)
                    message = json.loads(line)
                    proof.append(message)
                    if message["type"] == "prepared" and not cancelled:
                        budget = max(1, int((0.200 - (time.monotonic() - started)) * 1000))
                        child.stdin.write(f"S{budget}\n".encode())
                        child.stdin.flush()
        # Never kill an admitted owner outright. Its fixture finally retains and
        # reaps the holder on every normal/negative path before this wait ends.
        out, err = child.communicate(timeout=4)
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            buffered += chunk
        for line in buffered.splitlines():
            proof.append(json.loads(line))
        records = [json.loads(line) for line in (root / "kernel.jsonl").read_text().splitlines()]
        receipt = dict(argv=command, exit=child.returncode, elapsedMs=(time.monotonic()-started)*1000,
                       guardFired=guard_fired, proof=proof, records=records)
        (root / "stdout.txt").write_bytes(out)
        (root / "stderr.txt").write_bytes(err)
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        return receipt
    finally:
        primary = sys.exc_info()[1]
        cleanup_errors = []

        def attempt(action):
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)

        # Consume each owned endpoint once, even if another close refuses.
        endpoints = (read_fd, write_fd)
        read_fd = write_fd = None
        for fd in endpoints:
            if fd is not None:
                attempt(lambda fd=fd: os.close(fd))
        if child is not None:
            live = True
            try:
                live = child.poll() is None
            except BaseException as error:
                cleanup_errors.append(error)
            if live:
                # Cooperate with the retained fixture owner; never SIGKILL it
                # to disguise a descendant that its own finally must reap.
                attempt(lambda: child.send_signal(signal.SIGTERM))
                attempt(lambda: child.send_signal(signal.SIGCONT))
                attempt(lambda: child.wait(timeout=4))
            for stream in streams:
                if stream is not None:
                    attempt(stream.close)
        if cleanup_errors:
            if primary is not None:
                # Keep the original exception object/traceback authoritative,
                # including on system Python versions without add_note().
                try:
                    primary.fixture_cleanup_errors = cleanup_errors
                except BaseException:
                    pass
                try:
                    print("Gated fixture cleanup failures (not deallocation proof): " +
                          repr(cleanup_errors), file=sys.stderr)
                except BaseException:
                    pass  # Reporting must not replace the primary exception.
            elif len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            else:
                failure = RuntimeError("Gated fixture cleanup failed in multiple independent releases")
                failure.fixture_cleanup_errors = cleanup_errors
                raise failure from cleanup_errors[0]


class GatedChildOwnershipTests(unittest.TestCase):
    def check_case(self, scenario):
        source = Path(__file__).resolve().parents[2] / "scripts/phase5_recovery_process.py"
        output = os.environ.get("QMD_GATED_CHILD_OUTPUT")
        if output:
            root = Path(output) / scenario
        else:
            owned = tempfile.TemporaryDirectory(prefix="qmd-gated-regression-")
            self.addCleanup(owned.cleanup)
            root = Path(owned.name) / scenario
        result = run_case(source, scenario, root)
        detail = json.dumps(result)
        self.assertFalse(result["guardFired"], detail)
        self.assertTrue(any(r["kind"] == "boundary" for r in result["records"]), detail)
        if scenario.endswith("miss"):
            self.assertEqual(sum(r["kind"] == "discovery-omission" for r in result["records"]), 1, detail)
        if scenario == "live-miss":
            self.assertTrue(any(r["kind"] == "owned-live-at-zero" for r in result["records"]), detail)
        # Fixture cleanup is deliberately separate from product proof. It must
        # NOT repair the API receipt on the old-source RED path.
        self.assertEqual(result["exit"], 0, detail)
        self.assertEqual(result["records"][-1], dict(kind="fixture-cleanup", recovered=[]), detail)
        if scenario == "no-child":
            self.assertEqual([p["type"] for p in result["proof"]], ["prepared", "not-started"], detail)
            self.assertEqual(result["proof"][-1]["reason"], "parent-interrupted", detail)
        else:
            self.assertEqual([p["type"] for p in result["proof"]], ["prepared", "ready", "exit", "cleanup"], detail)
            proof = result["proof"][-1]
            self.assertTrue(proof["childReaped"], detail)
            self.assertTrue(proof["groupTerminated"] and proof["sessionTerminated"], detail)
            self.assertFalse(proof["containmentVerified"], detail)
        print(json.dumps(dict(evidence="gated-child-python", scenario=scenario,
                              sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              elapsedMs=result["elapsedMs"], productProof=result["proof"],
                              fixtureCleanup=result["records"][-1])))

    def test_01_no_child_kernel_echild(self):
        self.check_case("no-child")

    def test_02_live_owned_positive(self):
        self.check_case("live-control")

    def test_03_exiting_owned_omitted(self):
        self.check_case("exiting-miss")

    def test_04_live_owned_omitted(self):
        self.check_case("live-miss")

    def test_05_execution_lease_is_still_live_and_revocable(self):
        # Pure gate test: real pipe, no child/session, no fabricated API result.
        source = Path(__file__).resolve().parents[2] / "scripts/phase5_recovery_process.py"
        sys.path.insert(0, str(source.parent))
        scope = {"__name__": "gated_child_lease_unit"}
        exec(compile(source.read_bytes(), str(source), "exec"), scope)
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(read_fd, False)
            gate = scope["execution_allowed"]
            self.assertTrue(gate(read_fd, time.monotonic() + 0.100))
            self.assertFalse(gate(read_fd, time.monotonic() - 0.001))
            os.write(write_fd, b"T")
            self.assertFalse(gate(read_fd, time.monotonic() + 0.100))
            # The real gate consumes the cancellation byte while refusing.
            os.close(write_fd)
            write_fd = None
            self.assertFalse(gate(read_fd, time.monotonic() + 0.100))
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)


class GatedChildFixtureResourceTests(unittest.TestCase):
    source = Path(__file__).resolve().parents[2] / "scripts/phase5_recovery_process.py"

    def observe(self, label, fds, children=()):
        still_open = []
        for fd in fds:
            try:
                os.fstat(fd)  # Exact acquired descriptor, BEFORE any rescue.
                still_open.append(fd)
            except OSError as error:
                self.assertEqual(error.errno, errno.EBADF)
        state = dict(evidence="gated-fixture-python-before-independent-rescue", label=label,
                     ownedPipeFds=list(fds), openFds=still_open,
                     childrenReaped=all(child.returncode is not None for child in children),
                     streamsClosed=all(stream.closed for child in children
                                       for stream in (child.stdin, child.stdout, child.stderr) if stream is not None))
        print(json.dumps(state))
        return state

    def rescue(self, fds, children=()):
        # Exact retained custody. Every release is independent, even when OS
        # operations refuse; an assertion's primary error remains authoritative.
        primary = sys.exc_info()[1]
        failures, rescued = [], []
        # Call-owned diagnostics do not depend on a mutable primary exception.
        self.fixture_rescue_errors = failures
        def attempt(action):
            try:
                action()
            except BaseException as error:
                failures.append(error)
        def close_stream(child, name):
            stream = getattr(child, name)
            if stream is not None:
                stream.close()
        for child in children:
            live = True
            try:
                live = child.poll() is None  # Consuming operations belong ONLY to rescue.
            except BaseException as error:
                failures.append(error)
            if live:
                attempt(lambda: child.send_signal(signal.SIGTERM))
                attempt(lambda: child.send_signal(signal.SIGCONT))
                attempt(lambda: child.wait(timeout=4))
            for name in ("stdin", "stdout", "stderr"):
                attempt(lambda child=child, name=name: close_stream(child, name))
        for fd in fds:
            try:
                os.fstat(fd)
            except BaseException as error:
                if not isinstance(error, OSError) or error.errno != errno.EBADF:
                    failures.append(error)
            else:
                def close(fd=fd):
                    os.close(fd)
                    rescued.append(fd)
                attempt(close)
        # Construction, serialization and output are all fallible diagnostics.
        # Their failures remain secondary to earlier releases or the body.
        attempt(lambda: print(json.dumps(dict(evidence="gated-fixture-python-independent-rescue", rescuedFds=rescued,
                                              releaseFailures=[type(error).__name__ for error in failures]))))
        if failures:
            if primary is None:
                raise failures[0]
            attempt(lambda: setattr(primary, "fixture_rescue_errors", failures))

    def _rescue_reporting_control(self, mode, primary=None):
        from contextlib import ExitStack
        from unittest.mock import patch
        subject = GatedChildFixtureResourceTests("test_pre_rescue_observation_cannot_poll_or_reap")
        diagnostic = OSError("independent rescue " + mode + " diagnostic failure")
        acquisitions = []
        def refuse(*args, **kwargs):
            acquisitions.append(True)
            raise AssertionError("Reporting-only control attempted acquisition/release")
        thrown, threw = None, False
        with ExitStack() as stack:
            for obj, name in ((os, "pipe"), (os, "close"), (os, "kill"), (subprocess, "Popen")):
                stack.enter_context(patch.object(obj, name, refuse))
            reporter = stack.enter_context(patch("builtins.print", side_effect=diagnostic if mode == "report" else None))
            if mode == "serialization":
                stack.enter_context(patch.object(json, "dumps", side_effect=diagnostic))
            try:
                if primary is not None:
                    try:
                        raise primary
                    finally:
                        subject.rescue([])
                else:
                    subject.rescue([])
            except BaseException as error:
                thrown, threw = error, True
        self.assertEqual(acquisitions, [])
        return dict(threw=threw, thrown=thrown, diagnostic=diagnostic,
                    failures=getattr(subject, "fixture_rescue_errors", []), reports=reporter.call_count)

    def test_rescue_reporting_preserves_active_primary_when_print_fails(self):
        class FalsyPrimary(AssertionError):
            def __bool__(self):
                return False
        for primary in (AssertionError("unique print primary"), FalsyPrimary("false is still present")):
            result = self._rescue_reporting_control("report", primary)
            self.assertTrue(result["threw"])
            self.assertIs(result["thrown"], primary)
            self.assertEqual(result["failures"], [result["diagnostic"]])

    def test_rescue_reporting_preserves_active_primary_when_serialization_fails(self):
        primary = AssertionError("unique serialization primary")
        result = self._rescue_reporting_control("serialization", primary)
        self.assertTrue(result["threw"])
        self.assertIs(result["thrown"], primary)
        self.assertEqual(result["failures"], [result["diagnostic"]])

    def test_rescue_reporting_normal_report_preserves_primary_and_success(self):
        primary = AssertionError("unique normal-report primary")
        result = self._rescue_reporting_control("normal", primary)
        self.assertTrue(result["threw"])
        self.assertIs(result["thrown"], primary)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["reports"], 1)
        result = self._rescue_reporting_control("normal")
        self.assertFalse(result["threw"])
        self.assertEqual(result["reports"], 1)

    def test_rescue_reporting_print_only_failure_is_not_success(self):
        result = self._rescue_reporting_control("report")
        self.assertTrue(result["threw"])
        self.assertIs(result["thrown"], result["diagnostic"])

    def test_rescue_reporting_serialization_only_failure_is_not_success(self):
        result = self._rescue_reporting_control("serialization")
        self.assertTrue(result["threw"])
        self.assertIs(result["thrown"], result["diagnostic"])

    def test_rescue_reporting_attachment_failure_cannot_replace_nonmutable_primary(self):
        attachment = RuntimeError("primary refuses secondary-error attachment")
        class NonmutablePrimary(AssertionError):
            def __setattr__(self, name, value):
                if name == "fixture_rescue_errors":
                    raise attachment
                super().__setattr__(name, value)
        primary = NonmutablePrimary("exact attachment-resistant primary")
        result = self._rescue_reporting_control("report", primary)
        self.assertTrue(result["threw"])
        self.assertIs(result["thrown"], primary)
        self.assertEqual(result["failures"], [result["diagnostic"], attachment])

    def test_rescue_release_refusal_continues_other_owned_fd_before_independent_cleanup(self):
        from unittest.mock import patch
        owned = list(os.pipe())
        remaining, attempts = set(owned), []
        real_close = os.close
        refused = OSError(errno.EIO, "injected first rescue fd refusal: attempt is not deallocation")
        diagnostic = RuntimeError("later rescue print failure")
        subject = GatedChildFixtureResourceTests("test_pre_rescue_observation_cannot_poll_or_reap")
        def close(fd):
            attempts.append(fd)
            if fd == owned[0]:
                raise refused
            real_close(fd)
            remaining.remove(fd)  # Only descriptors still actually owned may be retried.
        try:
            before = self.observe("independent-rescue-fd-control-before-subject", owned)
            self.assertEqual(before["openFds"], owned)
            with patch.object(os, "close", close), patch("builtins.print", side_effect=diagnostic):
                with self.assertRaises(OSError) as failure:
                    subject.rescue(owned)
            state = self.observe("independent-rescue-fd-control-before-final-independent-cleanup", owned)
            self.assertIs(failure.exception, refused)
            self.assertEqual(subject.fixture_rescue_errors, [refused, diagnostic])
            self.assertEqual(attempts, owned)
            self.assertEqual(state["openFds"], [owned[0]])
            self.assertEqual(remaining, {owned[0]})
        finally:
            self.rescue(sorted(remaining))

    def capture_pipe(self, owned):
        real = os.pipe
        def pipe():
            pair = real()
            if not owned:  # Only the fixture pipe, not Popen's internal pipes.
                owned.extend(pair)
            return pair
        return pipe

    def test_pre_rescue_observation_cannot_poll_or_reap(self):
        # Synthetic no-child observer unit: poll is a consuming operation, not
        # proof that the candidate fixture itself already waited its child.
        class Unreaped:
            returncode = None
            stdin = stdout = stderr = None
            def poll(self):
                raise AssertionError("pre-rescue observation attempted consuming Popen.poll")
        state = self.observe("synthetic-nonconsuming-observer-unit", [], [Unreaped()])
        self.assertFalse(state["childrenReaped"])

    def test_launch_failure_closes_both_real_pipe_fds_and_preserves_error(self):
        from unittest.mock import patch  # Never add this import to supervisor startup.
        owned = []
        primary = OSError(errno.EAGAIN, "injected Popen failure before any child exists")
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            try:
                with patch.object(os, "pipe", self.capture_pipe(owned)), patch.object(subprocess, "Popen", side_effect=primary) as popen:
                    with self.assertRaises(OSError) as failure:
                        run_case(self.source, "live-miss", Path(root) / "case")
                state = self.observe("popen-no-child-failure", owned)
                self.assertIs(failure.exception, primary)
                self.assertEqual(popen.call_count, 1)
                self.assertEqual(len(owned), 2)
                self.assertEqual(state["openFds"], [])
            finally:
                self.rescue(owned)

    def test_pipe_allocation_failure_preserves_error_without_launch(self):
        from unittest.mock import patch
        primary = OSError(errno.EMFILE, "injected pipe allocation failure")
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            with patch.object(os, "pipe", side_effect=primary), patch.object(subprocess, "Popen") as popen:
                with self.assertRaises(OSError) as failure:
                    run_case(self.source, "live-miss", Path(root) / "case")
            self.assertIs(failure.exception, primary)
            popen.assert_not_called()

    def test_post_launch_setup_failure_reaps_owned_child_and_closes_fds_and_streams(self):
        from unittest.mock import patch
        owned, children = [], []
        real_popen = subprocess.Popen
        primary = OSError(errno.EINVAL, "injected set_blocking setup failure before S admission")
        def launch(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            try:
                with patch.object(os, "pipe", self.capture_pipe(owned)), patch.object(subprocess, "Popen", launch), patch.object(os, "set_blocking", side_effect=primary):
                    with self.assertRaises(OSError) as failure:
                        run_case(self.source, "live-miss", Path(root) / "case")
                state = self.observe("post-popen-setup-no-S-admission", owned, children)
                self.assertIs(failure.exception, primary)
                self.assertEqual(len(children), 1)
                self.assertEqual(state["openFds"], [])
                self.assertTrue(state["childrenReaped"] and state["streamsClosed"])
            finally:
                self.rescue(owned, children)

    def test_refused_fd_close_attempt_does_not_skip_other_endpoint_or_replace_primary(self):
        from unittest.mock import patch
        owned, attempts = [], []
        real_close = os.close
        primary = OSError(errno.EAGAIN, "injected Popen failure, no child")
        refused = OSError(errno.EIO, "mocked read endpoint close refusal: attempt only, not OS deallocation")
        def close(fd):
            if fd in owned:
                attempts.append(fd)
                if fd == owned[0]:
                    raise refused
            return real_close(fd)
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            try:
                with patch.object(os, "pipe", self.capture_pipe(owned)), patch.object(subprocess, "Popen", side_effect=primary), patch.object(os, "close", close):
                    with self.assertRaises(OSError) as failure:
                        run_case(self.source, "live-miss", Path(root) / "case")
                state = self.observe("mock-close-refusal-attempt-not-deallocation", owned)
                self.assertIs(failure.exception, primary)
                self.assertCountEqual(attempts, owned)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(state["openFds"], [owned[0]])
                self.assertEqual(getattr(primary, "fixture_cleanup_errors", None), [refused])
            finally:
                self.rescue(owned)

    def test_stream_release_failure_does_not_skip_other_streams_or_owned_wait(self):
        from unittest.mock import patch
        owned, children, streams, attempts = [], [], [], []
        real_popen = subprocess.Popen
        primary = OSError(errno.EINVAL, "injected post-launch setup refusal")
        refused = OSError(errno.EIO, "injected stdin close report failure after real close")
        class ObservedStream:
            def __init__(self, stream, name):
                self.stream, self.name = stream, name
            def close(self):
                attempts.append(self.name)
                self.stream.close()
                if self.name == "stdin":
                    raise refused
            @property
            def closed(self):
                return self.stream.closed
        def launch(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            original = (child.stdin, child.stdout, child.stderr)
            streams.append(original)
            child.stdin, child.stdout, child.stderr = [ObservedStream(stream, name) for stream, name in zip(original, ("stdin", "stdout", "stderr"))]
            return child
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            try:
                with patch.object(os, "pipe", self.capture_pipe(owned)), patch.object(subprocess, "Popen", launch), patch.object(os, "set_blocking", side_effect=primary):
                    with self.assertRaises(OSError) as failure:
                        run_case(self.source, "live-miss", Path(root) / "case")
                state = self.observe("stream-close-report-failure-after-real-close", owned, children)
                self.assertIs(failure.exception, primary)
                self.assertCountEqual(attempts, ["stdin", "stdout", "stderr"])
                self.assertEqual(state["openFds"], [])
                self.assertTrue(state["childrenReaped"] and state["streamsClosed"])
                self.assertEqual(getattr(primary, "fixture_cleanup_errors", None), [refused])
            finally:
                for child, original in zip(children, streams):
                    child.stdin, child.stdout, child.stderr = original
                self.rescue(owned, children)

    def test_cleanup_only_close_report_failure_is_not_success_or_double_close(self):
        from unittest.mock import patch
        owned, attempts, children = [], [], []
        real_close, real_popen = os.close, subprocess.Popen
        refused = OSError(errno.EIO, "injected read close report failure after actual close")
        def launch(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        def close(fd):
            if fd in owned:
                attempts.append(fd)
            result = real_close(fd)
            if owned and fd == owned[0]:
                raise refused
            return result
        with tempfile.TemporaryDirectory(prefix="qmd-gated-resource-") as root:
            try:
                with patch.object(os, "pipe", self.capture_pipe(owned)), patch.object(subprocess, "Popen", launch), patch.object(os, "close", close):
                    with self.assertRaises(OSError) as failure:
                        run_case(self.source, "no-child", Path(root) / "case")
                state = self.observe("cleanup-only-close-report-failure", owned, children)
                self.assertIs(failure.exception, refused)
                self.assertCountEqual(attempts, owned)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(state["openFds"], [])
                self.assertTrue(state["childrenReaped"] and state["streamsClosed"])
            finally:
                self.rescue(owned, children)


class GatedChildSupervisorRoutingTests(unittest.TestCase):
    """Execute this source's entry routing, never the native fixture body."""

    def entry_observation(self, arguments, module_name="__main__", interruption=None):
        import builtins
        import io
        from contextlib import ExitStack
        from types import ModuleType
        from unittest.mock import patch

        source = Path(__file__).resolve()
        source_bytes = source.read_bytes()
        module_code = compile(source_bytes, str(source), "exec")
        fixture_codes = [code for code in module_code.co_consts
                         if isinstance(code, type(module_code)) and code.co_name == "supervised_fixture"]
        self.assertEqual(len(fixture_codes), 1)
        fixture_code = fixture_codes[0]  # Identity from THIS complete compilation.
        module = ModuleType(module_name)
        scope = module.__dict__
        scope["__file__"] = str(source)
        imports, classes, entries, main_calls, attempts, controls = [], [], [], [], [], []

        class BoundaryStop(BaseException):
            pass

        boundary_stop = interruption if interruption is not None else BoundaryStop()
        main_stop = BoundaryStop()
        real_import, real_build_class = builtins.__import__, builtins.__build_class__

        def refuse(*args, **kwargs):
            attempts.append("refused")
            raise AssertionError("Routing control refused native/resource acquisition")

        def observe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if globals is scope:
                imports.append(name)
            if name.split(".")[0] in ("ctypes", "_ctypes", "mmap", "multiprocessing",
                                       "_multiprocessing", "sqlite3", "_sqlite3"):
                refuse()
            return real_import(name, globals, locals, fromlist, level)

        def observe_class(body, name, *bases, **kwargs):
            if body.__globals__ is scope:
                classes.append(name)
            return real_build_class(body, name, *bases, **kwargs)

        def driver_main(*args, **kwargs):
            main_calls.append((args, kwargs))
            raise main_stop  # Never let unittest execute the ownership methods.

        def trace(frame, event, arg):
            if frame.f_code is fixture_code:
                # A same-named function, another compilation or another globals
                # dictionary is NOT the source-bound boundary under test.
                self.assertIs(frame.f_globals, scope)
                self.assertIs(scope["supervised_fixture"].__code__, fixture_code)
                self.assertEqual(event, "call")
                entries.append(dict(arguments=dict(frame.f_locals), globals=tuple(scope),
                                    codeIdentity=True, globalsIdentity=True, event=event))
                raise boundary_stop  # Before even the fixture's first assertion.
            return trace

        targets = [(os, name) for name in (
            "fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen",
            "pipe", "pipe2", "open", "openpty", "dup", "dup2", "dup3", "fdopen",
            "read", "write", "close", "closerange", "kill", "killpg", "wait",
            "waitpid", "waitid", "wait3", "wait4", "setsid", "setpgid", "set_blocking",
            "eventfd", "memfd_create", "mkfifo", "mknod") if hasattr(os, name)]
        targets += [(os, name) for name in dir(os) if name.startswith(("exec", "spawn"))]
        targets += [(subprocess, "Popen"), (builtins, "open"), (io, "open"), (io, "FileIO")]
        targets += [(signal, name) for name in ("signal", "pthread_sigmask", "raise_signal",
                                               "alarm", "setitimer", "sigwait") if hasattr(signal, name)]
        for name, attributes in (
            ("_io", ("open", "FileIO")), ("_posixsubprocess", ("fork_exec",)),
            ("_thread", ("start_new_thread",)), ("threading", ("Thread",)),
            ("socket", ("socket", "socketpair")), ("_socket", ("socket", "socketpair")),
            ("ctypes", ("CDLL", "PyDLL")), ("mmap", ("mmap",)), ("sqlite3", ("connect",))):
            loaded = sys.modules.get(name)
            if loaded is not None:
                targets += [(loaded, attr) for attr in attributes if hasattr(loaded, attr)]
        previous_trace = sys.gettrace()
        thrown = None
        with ExitStack() as stack:
            for owner, name in targets:
                stack.enter_context(patch.object(owner, name, refuse))
            # Qualify each installed refusal before any subject code executes.
            for owner, name in targets:
                self.assertIs(getattr(owner, name), refuse)
                with self.assertRaisesRegex(AssertionError, "Routing control refused"):
                    getattr(owner, name)()
                controls.append(owner.__name__ + "." + name)
            self.assertEqual(len(attempts), len(controls))
            attempts.clear()
            stack.enter_context(patch.object(builtins, "__import__", observe_import))
            stack.enter_context(patch.object(builtins, "__build_class__", observe_class))
            stack.enter_context(patch.object(unittest, "main", driver_main))
            stack.enter_context(patch.object(sys, "argv", [str(source)] + list(arguments)))
            try:
                sys.settrace(trace)
                exec(module_code, scope)
            except BaseException as error:
                thrown = error
            finally:
                sys.settrace(previous_trace)
        self.assertEqual(attempts, [], "Subject attempted a native/resource operation")
        summary = dict(sourceSha256=hashlib.sha256(source_bytes).hexdigest(), imports=imports,
                       classes=classes, argv=[str(source)] + list(arguments), moduleName=module_name,
                       entries=[dict(event=e["event"], codeIdentity=e["codeIdentity"],
                                     globalsIdentity=e["globalsIdentity"], globals=e["globals"])
                                for e in entries], fixtureBodyExecuted=False,
                       acquisitionAttempts=attempts, qualifiedRefusals=controls,
                       exceptionType=type(thrown).__name__ if thrown is not None else None)
        if not hasattr(self, "routing_observations"):
            self.routing_observations = []
        self.routing_observations.append(summary)
        return dict(module=module, entries=entries, imports=imports, classes=classes,
                    mainCalls=main_calls, thrown=thrown, boundaryStop=boundary_stop,
                    mainStop=main_stop, controls=controls)

    def supervisor_arguments(self):
        source = Path(__file__).resolve().parents[2] / "scripts/phase5_recovery_process.py"
        root = Path(__file__).parent / "routing-must-not-acquire"
        return ["--supervisor", str(source), "live-miss", str(root), "037", "opaque-config", "ignored-extra"]

    def test_supervisor_dispatch_precedes_test_driver_work(self):
        arguments = self.supervisor_arguments()
        result = self.entry_observation(arguments)
        self.assertIs(result["thrown"], result["boundaryStop"])
        self.assertEqual(len(result["entries"]), 1)
        entry = result["entries"][0]
        self.assertEqual(entry["arguments"], dict(source=Path(arguments[1]), scenario=arguments[2],
                                                root=Path(arguments[3]), proof_fd=37, config=arguments[5]))
        self.assertEqual(result["imports"], ["__future__", "errno", "hashlib", "json", "os",
                                              "pathlib", "signal", "sys", "time"])
        self.assertEqual(result["classes"], [])
        for deferred in ("select", "subprocess", "tempfile", "traceback", "unittest", "run_case"):
            self.assertNotIn(deferred, entry["globals"])
        for required in ("errno", "hashlib", "json", "os", "Path", "signal", "sys", "time", "SCENARIOS"):
            self.assertIn(required, entry["globals"])
        for refusal in ("os.fork", "os.pipe", "os.dup2", "os.kill", "os.waitpid",
                        "subprocess.Popen", "signal.signal", "io.open"):
            self.assertIn(refusal, result["controls"])
        self.assertEqual(result["mainCalls"], [])

    def test_supervisor_boundary_exceptions_propagate_unchanged(self):
        # These are boundary interruptions, NOT executions of the native body.
        for primary in (AssertionError("exact routing primary"), SystemExit(23)):
            with self.subTest(primary=type(primary).__name__):
                result = self.entry_observation(self.supervisor_arguments(), interruption=primary)
                self.assertIs(result["thrown"], primary)
                self.assertEqual(len(result["entries"]), 1)
                self.assertEqual(result["mainCalls"], [])

    def test_malformed_supervisor_arguments_fail_before_fixture_entry(self):
        arguments = self.supervisor_arguments()
        cases = [(arguments[:length], IndexError) for length in range(1, 6)]
        cases.append((arguments[:4] + ["not-an-int", arguments[5]], ValueError))
        for argv, expected in cases:
            with self.subTest(argv=argv):
                result = self.entry_observation(argv)
                self.assertIsInstance(result["thrown"], expected)
                self.assertEqual(result["entries"], [])
                self.assertEqual(result["classes"], [])
                self.assertEqual(result["mainCalls"], [])

    def test_import_with_supervisor_argv_preserves_original_test_discovery(self):
        result = self.entry_observation(self.supervisor_arguments(), module_name="gated_routing_import")
        self.assertIsNone(result["thrown"])
        self.assertEqual(result["entries"], [])
        self.assertEqual(result["mainCalls"], [])
        expected = {
            "GatedChildOwnershipTests": (
                "test_01_no_child_kernel_echild", "test_02_live_owned_positive",
                "test_03_exiting_owned_omitted", "test_04_live_owned_omitted",
                "test_05_execution_lease_is_still_live_and_revocable"),
            "GatedChildFixtureResourceTests": (
                "test_cleanup_only_close_report_failure_is_not_success_or_double_close",
                "test_launch_failure_closes_both_real_pipe_fds_and_preserves_error",
                "test_pipe_allocation_failure_preserves_error_without_launch",
                "test_post_launch_setup_failure_reaps_owned_child_and_closes_fds_and_streams",
                "test_pre_rescue_observation_cannot_poll_or_reap",
                "test_refused_fd_close_attempt_does_not_skip_other_endpoint_or_replace_primary",
                "test_rescue_release_refusal_continues_other_owned_fd_before_independent_cleanup",
                "test_rescue_reporting_attachment_failure_cannot_replace_nonmutable_primary",
                "test_rescue_reporting_normal_report_preserves_primary_and_success",
                "test_rescue_reporting_preserves_active_primary_when_print_fails",
                "test_rescue_reporting_preserves_active_primary_when_serialization_fails",
                "test_rescue_reporting_print_only_failure_is_not_success",
                "test_rescue_reporting_serialization_only_failure_is_not_success",
                "test_stream_release_failure_does_not_skip_other_streams_or_owned_wait"),
        }
        discovered = []
        for name, methods in expected.items():
            cls = getattr(result["module"], name)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
            actual = [test._testMethodName for test in suite]  # Discovery ONLY.
            self.assertEqual(actual, sorted(methods))
            discovered.extend(actual)
        self.assertEqual(len(discovered), 19)
        self.assertEqual(result["classes"], ["GatedChildOwnershipTests", "GatedChildFixtureResourceTests",
                                             "GatedChildSupervisorRoutingTests"])

    def test_other_cli_modes_still_dispatch_to_unittest(self):
        for arguments in ([], ["--unknown-option"], ["--supervisor-suffix"]):
            with self.subTest(arguments=arguments):
                result = self.entry_observation(arguments)
                self.assertIs(result["thrown"], result["mainStop"])
                self.assertEqual(result["entries"], [])
                self.assertEqual(result["mainCalls"], [((), {"verbosity": 2})])
                self.assertIn("GatedChildOwnershipTests", result["classes"])
                self.assertIn("GatedChildFixtureResourceTests", result["classes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
