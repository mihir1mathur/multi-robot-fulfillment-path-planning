"""A small, typed HTTP client for the fulfillment API.

Everything the dashboard does goes through this client. It:

* sets a real connect/read timeout on every call (no silent hangs),
* turns transport failures into ``ApiUnavailable`` and 4xx/5xx into
  ``ApiError`` with the service's own ``message`` when there is one,
* never puts a password in a URL, a log line, or an exception message,
* returns plain ``dict`` / ``list`` payloads - parsing into view rows is the
  job of ``view_model.py``.

It depends only on ``httpx`` (already a project dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 8.0


class ApiUnavailable(RuntimeError):
    """The service could not be reached at all (connection refused, DNS, timeout)."""


class ApiError(RuntimeError):
    """The service answered with a 4xx / 5xx."""

    def __init__(self, status_code: int, message: str, error_code: str = "") -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.error_code = error_code


@dataclass
class FulfillmentClient:
    """Stateless-ish wrapper around the FastAPI service.

    A ``token`` (from :meth:`login`) is remembered and sent as a bearer header
    on every subsequent call. Construct with an ``httpx.Client`` in tests to
    inject a mock transport.
    """

    base_url: str = "http://127.0.0.1:8000"
    timeout: float = DEFAULT_TIMEOUT
    token: Optional[str] = None
    _client: Optional[httpx.Client] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 5.0)),
        )

    def _headers(self) -> Dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        data: Any = None,
        params: Any = None,
        auth_required: bool = True,
    ) -> Any:
        owns_client = self._client is None
        client = self._http()
        try:
            response = client.request(
                method,
                path,
                json=json,
                data=data,
                params=params,
                headers=self._headers() if auth_required else None,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ApiUnavailable(
                f"cannot reach the API at {self.base_url} - is it running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiUnavailable(
                f"the API at {self.base_url} did not respond within {self.timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:  # any other transport-level problem
            raise ApiUnavailable(f"HTTP transport error talking to {self.base_url}") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code >= 400:
            raise _as_api_error(response)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                response.status_code, "the API returned a response that was not JSON"
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health", auth_required=False)

    def ready(self) -> Dict[str, Any]:
        try:
            return self._request("GET", "/ready", auth_required=False)
        except ApiError as exc:
            if exc.status_code == 503:
                return {"status": "not_ready", "checks": {"database": "unavailable"}}
            raise

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def login(self, username: str, password: str) -> str:
        payload = self._request(
            "POST",
            "/auth/token",
            data={"username": username, "password": password},
            auth_required=False,
        )
        self.token = payload["access_token"]
        return self.token

    # ------------------------------------------------------------------
    # Robots / tasks (persisted state)
    # ------------------------------------------------------------------
    def list_robots(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"status": status} if status else None
        return self._request("GET", "/robots", params=params)["robots"]

    def create_robot(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/robots", json=body)

    def update_robot(self, robot_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PATCH", f"/robots/{robot_id}", json=body)

    def delete_robot(self, robot_id: str) -> None:
        self._request("DELETE", f"/robots/{robot_id}")

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"status": status} if status else None
        return self._request("GET", "/tasks", params=params)["tasks"]

    def create_task(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/tasks", json=body)

    def update_task(self, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PATCH", f"/tasks/{task_id}", json=body)

    def delete_task(self, task_id: str) -> None:
        self._request("DELETE", f"/tasks/{task_id}")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def plan_path(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/planning/path", json=body)

    def run_allocation(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/allocation/run", json=body)

    def plan_coordination(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/coordination/plan", json=body)

    def recover_obstacle(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/recovery/obstacle", json=body)

    def recover_robot_failure(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/recovery/robot-failure", json=body)


def _as_api_error(response: httpx.Response) -> ApiError:
    message = f"the API returned {response.status_code}"
    error_code = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            message = body.get("message") or body.get("detail") or message
            error_code = body.get("error", "") or ""
    except ValueError:
        text = (response.text or "").strip()
        if text:
            message = text[:300]
    return ApiError(response.status_code, message, error_code)
