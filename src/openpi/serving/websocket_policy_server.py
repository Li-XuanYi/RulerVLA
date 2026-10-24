import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies.policy import AccelFeedbackConfig
from openpi.uncertainty.accel_runtime import AccelRuntime

logger = logging.getLogger(__name__)

# Sent by a rollout client on the first request of each episode.  The CUSUM is a
# causal random walk, so without a per-episode reset it accumulates across
# episode boundaries and fires on ordinary drift; the boundary is a protocol
# obligation, not an optimisation.
EPISODE_START_KEY = "episode_start"


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.

    With ``accel_runtime`` set, each request additionally scores the model-space
    denoising path and advances a causal CUSUM; the selected feedback strategy
    decides what to do after an alarm (by default, shorten the action chunk).
    The runtime holds a single monitor, so it assumes one rollout client per
    server: several concurrent clients would interleave their episodes.  Leave
    ``accel_runtime`` unset and the request path is byte-for-byte the baseline.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        accel_runtime: AccelRuntime | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # Optional application-layer monitor.  It is deliberately disabled by
        # default so the ordinary WebSocket inference path remains identical
        # to the existing LIBERO/Baseline server.
        self._accel_runtime = accel_runtime
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        if self._accel_runtime is not None:
            # A connection is the only episode boundary we can rely on: a client
            # that never sends the marker would otherwise let the CUSUM walk
            # across episodes, so start every connection from a clean state.
            # This deliberately consumes no episode number -- the monitor is
            # reset, but no rollout began, and numbering a phantom episode here
            # would shift every subsequent stream against the outcome file.
            self._accel_runtime.reset_connection()

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                if self._accel_runtime is None:
                    action = self._policy.infer(obs)
                else:
                    if not hasattr(self._policy, "infer_with_accel"):
                        raise NotImplementedError("the served policy does not expose infer_with_accel")
                    runtime = self._accel_runtime
                    # Consumed here, before the policy sees the observation and
                    # upstream of both infer_with_accel and the feedback variant,
                    # so the marker can never reach the input transforms or the
                    # JIT pytree.
                    marker = obs.pop(EPISODE_START_KEY, None)
                    if marker:
                        # Any truthy value opens an episode; the LIBERO client
                        # sends the episode id itself so the record can be
                        # joined against its outcomes without relying on order.
                        runtime.start_episode(None if marker is True else str(marker))
                    # Accel is a read-only diagnostic.  Keep the action chunk
                    # from the same FM trajectory, but do not transmit the
                    # potentially large intermediate trajectory over the wire.
                    accel_kwargs = runtime.call_kwargs()
                    if runtime.capturing_context:
                        accel_kwargs["capture_context"] = True
                    # The policy owns the monitor so that one observation yields
                    # exactly one causal decision, taken on the first trajectory
                    # of the request rather than on a resample.
                    accel_kwargs["accel_monitor"] = runtime.monitor
                    feedback_kwargs = runtime.feedback_kwargs()
                    if feedback_kwargs is None:
                        action = self._policy.infer_with_accel(obs, **accel_kwargs)
                    else:
                        if not hasattr(self._policy, "infer_with_accel_feedback"):
                            raise NotImplementedError("the served policy does not expose Accel feedback")
                        action = self._policy.infer_with_accel_feedback(
                            obs,
                            feedback=AccelFeedbackConfig(**feedback_kwargs),
                            **accel_kwargs,
                        )
                    # Pool the reference scale and record, then drop the
                    # trajectory from the response: it is large and no rollout
                    # client consumes it.
                    trajectory = action.pop("fm_trajectory", None)
                    # Popped rather than transmitted: every denoise depth of one
                    # request is ~T-1 floats, which adds up to tens of megabytes
                    # across a full rollout run that no client consumes.
                    prefix_table = action.pop("accel_prefix_table", None)
                    prefix_table_full = action.pop("accel_prefix_table_full", None)
                    # The captured observation is only needed offline, where the
                    # resample ground truth is computed; it is ~300 KB per
                    # decision and no rollout client consumes it.
                    fm_context = action.pop("fm_context", None)
                    monitor_info = action.get("accel_monitor")
                    if monitor_info is None:
                        # Resampling strategies return the replacement inference,
                        # which carries no monitor block; the decision still
                        # records the score and CUSUM state the alarm used.
                        monitor_info = action.get("feedback")
                    record = None
                    if trajectory is not None:
                        # The monitor's score when there is one, not a resample's:
                        # the record has to describe the decision that was taken.
                        score = (
                            action.get("accel")
                            if monitor_info is None
                            else monitor_info.get("score", action.get("accel"))
                        )
                        record = runtime.observe(
                            trajectory["x_t"],
                            v_t=trajectory.get("v_t"),
                            score=score,
                            monitor_info=monitor_info,
                            prefix_table=prefix_table,
                            prefix_table_full=prefix_table_full,
                            context=fm_context,
                        )
                    if record is None:
                        monitor_info = None
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                # Feedback strategies may intentionally commit to fewer actions
                # than the caller's default replan window.  Expose the requested
                # execution horizon so rollout clients re-observe in time.  The
                # adaptive strategy reports one step count per request, so it is
                # transmitted even when the policy kept the full window.
                feedback = action.get("feedback", {})
                execute_steps = feedback.get("execute_steps")
                if execute_steps is None and feedback.get("triggered") and feedback.get("strategy") == "shorten":
                    execute_steps = len(action["actions"])
                if execute_steps is not None and "actions" in action:
                    action["execution_horizon"] = int(execute_steps)
                if "accel" in action:
                    monitor_info = monitor_info or {}
                    cusum = monitor_info.get("cusum")
                    logger.info(
                        "Accel score=%.6f cusum=%s onset=%s index=%s triggered=%s strategy=%s "
                        "execution_horizon=%s std_ready=%s",
                        float(action["accel"]),
                        "n/a" if cusum is None else f"{float(cusum):.4f}",
                        monitor_info.get("onset", "n/a"),
                        monitor_info.get("index", "n/a"),
                        feedback.get("triggered", False),
                        feedback.get("strategy", "monitor-only"),
                        action.get("execution_horizon", "default"),
                        self._accel_runtime is not None and not self._accel_runtime.calibrating,
                    )
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                if self._accel_runtime is not None:
                    # Persist whatever the client already captured: a reconnect
                    # resumes into the same episode file, but a run that simply
                    # ended must not lose its last buffered decisions.
                    self._accel_runtime.flush_context()
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
