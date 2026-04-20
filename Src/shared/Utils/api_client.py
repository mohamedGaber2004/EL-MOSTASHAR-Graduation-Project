import httpx
import logging
from typing import Any, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class ServiceCommunicator:
    """
    A professional HTTP client wrapper for inter-service communication.
    Provides standard timeouts, structured JSON handling, and automatic 
    exponential backoff retries for transient networking failures.
    """

    def __init__(self, service_name: str, base_url: str = "", timeout: float = 30.0):
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # Decorator for retrying on connection errors or 5xx server errors
    def _create_retry_decorator(self):
        return retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=15),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError)),
            before_sleep=lambda retry_state: logger.warning(
                f"[{self.service_name}] Request failed (attempt {retry_state.attempt_number}). Retrying..."
            )
        )

    def post(self, endpoint: str, json_payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        """Synchronous POST request with automatic retries."""
        url = f"{self.base_url}{endpoint}"
        
        @self._create_retry_decorator()
        def _do_post():
            with httpx.Client(timeout=self.timeout) as client:
                logger.info(f"[{self.service_name}] POST -> {url}")
                response = client.post(url, json=json_payload, headers=headers)
                response.raise_for_status()
                return response.json()
        
        try:
            return _do_post()
        except Exception as e:
            logger.error(f"[{self.service_name}] Sync POST to {url} permanently failed: {str(e)}")
            raise

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        """Synchronous GET request with automatic retries."""
        url = f"{self.base_url}{endpoint}"
        
        @self._create_retry_decorator()
        def _do_get():
            with httpx.Client(timeout=self.timeout) as client:
                logger.info(f"[{self.service_name}] GET -> {url}")
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
                
        try:
            return _do_get()
        except Exception as e:
            logger.error(f"[{self.service_name}] Sync GET to {url} permanently failed: {str(e)}")
            raise

    async def async_post(self, endpoint: str, json_payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        """Asynchronous POST request with automatic retries."""
        url = f"{self.base_url}{endpoint}"
        
        @self._create_retry_decorator()
        async def _do_async_post():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"[{self.service_name}] Async POST -> {url}")
                response = await client.post(url, json=json_payload, headers=headers)
                response.raise_for_status()
                return response.json()
                
        try:
            return await _do_async_post()
        except Exception as e:
            logger.error(f"[{self.service_name}] Async POST to {url} permanently failed: {str(e)}")
            raise

    async def async_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        """Asynchronous GET request with automatic retries."""
        url = f"{self.base_url}{endpoint}"
        
        @self._create_retry_decorator()
        async def _do_async_get():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"[{self.service_name}] Async GET -> {url}")
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
                
        try:
            return await _do_async_get()
        except Exception as e:
            logger.error(f"[{self.service_name}] Async GET to {url} permanently failed: {str(e)}")
            raise
