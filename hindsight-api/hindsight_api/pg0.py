import asyncio
import logging
import os
import json

from pg0 import Pg0

logger = logging.getLogger(__name__)

# #region agent log
import pathlib
_DEBUG_LOG_PATH = str(pathlib.Path(__file__).parent.parent.parent / "debug.log")
def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    try:
        import time
        entry = {"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000), "sessionId": "debug-session"}
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except: pass
# #endregion

DEFAULT_USERNAME = "hindsight"
DEFAULT_PASSWORD = "hindsight"
DEFAULT_DATABASE = "hindsight"


class EmbeddedPostgres:
    """Manages an embedded PostgreSQL server instance using pg0-embedded."""

    def __init__(
        self,
        port: int | None = None,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
        name: str = "hindsight",
        **kwargs,
    ):
        self.port = port  # None means pg0 will auto-assign
        self.username = username
        self.password = password
        self.database = database
        self.name = name
        self._pg0: Pg0 | None = None

    def _get_pg0(self) -> Pg0:
        if self._pg0 is None:
            kwargs = {
                "name": self.name,
                "username": self.username,
                "password": self.password,
                "database": self.database,
                # Try to configure PostgreSQL without timezone validation
                "config": {
                    "timezone": "UTC",
                    "log_timezone": "UTC",
                },
            }
            # Only set port if explicitly specified
            if self.port is not None:
                kwargs["port"] = self.port
            # #region agent log
            _debug_log("I", "pg0.py:get_pg0", "Creating Pg0 instance with config", {"kwargs_keys": list(kwargs.keys()), "config": kwargs.get("config")})
            # #endregion
            self._pg0 = Pg0(**kwargs)  # type: ignore[invalid-argument-type] - dict kwargs
        return self._pg0

    def _ensure_timezone_data(self):
        """Create minimal timezone data if system lacks it."""
        import struct
        import tempfile
        
        # First check if system already has timezone data
        system_paths = ["/usr/share/zoneinfo/UTC", "/usr/lib/zoneinfo/UTC", "/etc/zoneinfo/UTC"]
        for path in system_paths:
            if os.path.exists(path):
                # #region agent log
                _debug_log("G", "pg0.py:ensure_tz", "System UTC file exists", {"path": path})
                # #endregion
                return
        
        # Create timezone data in a user-writable location
        # Use ~/.local/share/zoneinfo or temp directory
        home = os.environ.get("HOME", tempfile.gettempdir())
        user_tzdir = os.path.join(home, ".local", "share", "zoneinfo")
        
        # #region agent log
        _debug_log("G", "pg0.py:ensure_tz", "Creating user timezone dir", {"user_tzdir": user_tzdir})
        # #endregion
        
        try:
            os.makedirs(user_tzdir, exist_ok=True)
            os.makedirs(os.path.join(user_tzdir, "Etc"), exist_ok=True)
            
            # Minimal TZif2 file for UTC (no DST, offset 0)
            tzif_data = (
                b'TZif2' + b'\x00' * 15 +  # magic + version + reserved
                b'\x00' * 24 +  # v1 counts (all zeros - skip to v2)
                b'TZif2' + b'\x00' * 15 +  # v2 magic + version + reserved  
                struct.pack('>6I', 0, 0, 0, 1, 1, 4) +  # v2 counts
                struct.pack('>lBB', 0, 0, 0) +  # ttinfo: offset=0, dst=0, abbr_idx=0
                b'UTC\x00' +  # timezone abbreviation
                b'\n<UTC>0\n'  # POSIX TZ string footer
            )
            
            utc_path = os.path.join(user_tzdir, "UTC")
            etc_utc_path = os.path.join(user_tzdir, "Etc", "UTC")
            
            with open(utc_path, 'wb') as f:
                f.write(tzif_data)
            with open(etc_utc_path, 'wb') as f:
                f.write(tzif_data)
            
            # Set TZDIR so PostgreSQL can find our timezone files
            os.environ["TZDIR"] = user_tzdir
            
            # #region agent log
            _debug_log("G", "pg0.py:ensure_tz", "Created timezone files and set TZDIR", {
                "utc_path": utc_path, 
                "etc_utc_path": etc_utc_path,
                "TZDIR": user_tzdir
            })
            # #endregion
        except Exception as e:
            # #region agent log
            _debug_log("G", "pg0.py:ensure_tz", "Error creating user timezone", {"error": str(e), "error_type": type(e).__name__})
            # #endregion

    async def start(self, max_retries: int = 5, retry_delay: float = 4.0) -> str:
        """Start the PostgreSQL server with retry logic."""
        port_info = f"port={self.port}" if self.port else "port=auto"
        logger.info(f"Starting embedded PostgreSQL (name={self.name}, {port_info})...")

        # Ensure TZ is set for embedded PostgreSQL timezone support
        # pg0-embedded requires TZ to be set for proper timezone handling
        # Use POSIX format which doesn't require timezone database files
        if "TZ" not in os.environ:
            os.environ["TZ"] = "UTC0"  # POSIX format: UTC with 0 offset
            # #region agent log
            _debug_log("FIX", "pg0.py:start:tz_set", "Set TZ environment variable", {"TZ": "UTC0"})
            # #endregion
        # Also set PGTZ to help PostgreSQL
        if "PGTZ" not in os.environ:
            os.environ["PGTZ"] = "UTC0"
            # #region agent log
            _debug_log("FIX", "pg0.py:start:pgtz_set", "Set PGTZ environment variable", {"PGTZ": "UTC0"})
            # #endregion
        
        # Try to create system timezone data if missing (for pg0 compatibility)
        self._ensure_timezone_data()

        # #region agent log
        _debug_log("A", "pg0.py:start:entry", "Environment vars for timezone", {
            "TZ": os.environ.get("TZ", "<not set>"),
            "PGTZ": os.environ.get("PGTZ", "<not set>"),
            "LC_ALL": os.environ.get("LC_ALL", "<not set>"),
            "LANG": os.environ.get("LANG", "<not set>"),
            "TZDIR": os.environ.get("TZDIR", "<not set>"),
        })
        # #endregion

        pg0 = self._get_pg0()
        
        # #region agent log
        # Try to find pg0's installation directory
        import glob
        home = os.environ.get("HOME", "/tmp")
        # Search various possible locations
        search_results = {}
        for pattern in [
            f"{home}/.local/share/pg0*",
            f"{home}/.cache/pg0*",
            f"{home}/.pg0*",
            f"{home}/pg0*",
            "/tmp/pg0*",
            "/tmp/.pg0*",
            f"{home}/.local/share/postgresql*",
            f"{home}/.cargo/*pg0*",
        ]:
            matches = glob.glob(pattern)
            if matches:
                search_results[pattern] = matches[:5]
        
        # Also find pg0 package location
        try:
            import pg0 as pg0_module
            pg0_file = pg0_module.__file__
            pg0_dir = os.path.dirname(pg0_file) if pg0_file else None
        except:
            pg0_file = None
            pg0_dir = None
            
        _debug_log("H", "pg0.py:start:pg0_paths", "Looking for pg0 installation", {
            "home": home,
            "search_results": search_results,
            "pg0_module_file": pg0_file,
            "pg0_module_dir": pg0_dir,
        })
        # #endregion
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # #region agent log
                # Check pg0 object attributes for installation path
                pg0_attrs = {k: str(getattr(pg0, k, None))[:200] for k in dir(pg0) if not k.startswith('_')}
                _debug_log("B", "pg0.py:start:before_pg0_start", f"Attempt {attempt}", {
                    "attempt": attempt, 
                    "pg0_name": self.name,
                    "pg0_attrs": pg0_attrs
                })
                # #endregion
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, pg0.start)
                # Get URI from pg0 (includes auto-assigned port)
                uri = info.uri
                # #region agent log
                _debug_log("B", "pg0.py:start:success", "pg0 started successfully", {"uri": uri})
                # #endregion
                logger.info(f"PostgreSQL started: {uri}")
                return uri
            except Exception as e:
                last_error = str(e)
                # #region agent log
                _debug_log("C", "pg0.py:start:exception", f"pg0 start failed attempt {attempt}", {"error": last_error, "error_type": type(e).__name__, "attempt": attempt})
                # #endregion
                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.debug(f"pg0 start attempt {attempt}/{max_retries} failed: {last_error}")
                    logger.debug(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.debug(f"pg0 start attempt {attempt}/{max_retries} failed: {last_error}")

        raise RuntimeError(
            f"Failed to start embedded PostgreSQL after {max_retries} attempts. Last error: {last_error}"
        )

    async def stop(self) -> None:
        """Stop the PostgreSQL server."""
        pg0 = self._get_pg0()
        logger.info(f"Stopping embedded PostgreSQL (name: {self.name})...")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, pg0.stop)
            logger.info("Embedded PostgreSQL stopped")
        except Exception as e:
            if "not running" in str(e).lower():
                return
            raise RuntimeError(f"Failed to stop PostgreSQL: {e}")

    async def get_uri(self) -> str:
        """Get the connection URI for the PostgreSQL server."""
        pg0 = self._get_pg0()
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, pg0.info)
        return info.uri

    async def is_running(self) -> bool:
        """Check if the PostgreSQL server is currently running."""
        try:
            pg0 = self._get_pg0()
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, pg0.info)
            return info is not None and info.running
        except Exception:
            return False

    async def ensure_running(self) -> str:
        """Ensure the PostgreSQL server is running, starting it if needed."""
        if await self.is_running():
            return await self.get_uri()
        return await self.start()


_default_instance: EmbeddedPostgres | None = None


def get_embedded_postgres() -> EmbeddedPostgres:
    """Get or create the default EmbeddedPostgres instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = EmbeddedPostgres()
    return _default_instance


async def start_embedded_postgres() -> str:
    """Quick start function for embedded PostgreSQL."""
    return await get_embedded_postgres().ensure_running()


async def stop_embedded_postgres() -> None:
    """Stop the default embedded PostgreSQL instance."""
    global _default_instance
    if _default_instance:
        await _default_instance.stop()


def parse_pg0_url(db_url: str) -> tuple[bool, str | None, int | None]:
    """
    Parse a database URL and check if it's a pg0:// embedded database URL.

    Supports:
    - "pg0" -> default instance "hindsight"
    - "pg0://instance-name" -> named instance
    - "pg0://instance-name:port" -> named instance with explicit port
    - Any other URL (e.g., postgresql://) -> not a pg0 URL

    Args:
        db_url: The database URL to parse

    Returns:
        Tuple of (is_pg0, instance_name, port)
        - is_pg0: True if this is a pg0 URL
        - instance_name: The instance name (or None if not pg0)
        - port: The explicit port (or None for auto-assign)
    """
    if db_url == "pg0":
        return True, "hindsight", None

    if db_url.startswith("pg0://"):
        url_part = db_url[6:]  # Remove "pg0://"
        if ":" in url_part:
            instance_name, port_str = url_part.rsplit(":", 1)
            return True, instance_name or "hindsight", int(port_str)
        else:
            return True, url_part or "hindsight", None

    return False, None, None


async def resolve_database_url(db_url: str) -> str:
    """
    Resolve a database URL, handling pg0:// embedded database URLs.

    If the URL is a pg0:// URL, starts the embedded PostgreSQL and returns
    the actual postgresql:// connection URL. Otherwise, returns the URL unchanged.

    Args:
        db_url: Database URL (pg0://, pg0, or postgresql://)

    Returns:
        The resolved postgresql:// connection URL
    """
    is_pg0, instance_name, port = parse_pg0_url(db_url)
    if is_pg0:
        pg0 = EmbeddedPostgres(name=instance_name, port=port)
        return await pg0.ensure_running()
    return db_url
