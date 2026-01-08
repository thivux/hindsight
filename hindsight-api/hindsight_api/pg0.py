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
            }
            # Only set port if explicitly specified
            if self.port is not None:
                kwargs["port"] = self.port
            # #region agent log
            _debug_log("I", "pg0.py:get_pg0", "Creating Pg0 instance", {"kwargs_keys": list(kwargs.keys())})
            # #endregion
            self._pg0 = Pg0(**kwargs)  # type: ignore[invalid-argument-type] - dict kwargs
        return self._pg0

    def _ensure_timezone_data(self):
        """Create minimal timezone data if system lacks it."""
        import struct
        import tempfile
        import glob
        
        # Proper TZif2 file for UTC timezone
        # See: https://www.rfc-editor.org/rfc/rfc8536
        
        # V1 header (44 bytes total)
        v1_header = (
            b'TZif' +                      # 4 bytes: magic
            b'2' +                         # 1 byte: version
            b'\x00' * 15 +                 # 15 bytes: reserved
            struct.pack('>6I',             # 24 bytes: counts (all zeros for v1 - skip to v2)
                0,  # ttisutcnt
                0,  # ttisstdcnt  
                0,  # leapcnt
                0,  # timecnt
                0,  # typecnt
                0,  # charcnt
            )
        )
        
        # V2 header (44 bytes)
        v2_header = (
            b'TZif' +                      # 4 bytes: magic
            b'2' +                         # 1 byte: version
            b'\x00' * 15 +                 # 15 bytes: reserved
            struct.pack('>6I',             # 24 bytes: counts
                0,  # ttisutcnt - number of UT/local indicators
                0,  # ttisstdcnt - number of standard/wall indicators
                0,  # leapcnt - number of leap second records
                0,  # timecnt - number of transition times
                1,  # typecnt - number of local time type records (need at least 1!)
                4,  # charcnt - total chars in abbreviation strings ("UTC\0")
            )
        )
        
        # V2 data
        # ttinfo record (6 bytes): utoff (4 bytes signed) + dst (1 byte) + idx (1 byte)
        v2_ttinfo = struct.pack('>iBB', 0, 0, 0)  # offset=0, dst=false, abbr_idx=0
        v2_abbrev = b'UTC\x00'  # timezone abbreviation with null terminator
        
        # POSIX TZ footer (starts with newline, ends with newline)
        footer = b'\nUTC0\n'
        
        tzif_data = v1_header + v2_header + v2_ttinfo + v2_abbrev + footer
        
        def write_tz_files(base_dir):
            """Write UTC timezone files to a directory, overwriting existing."""
            try:
                os.makedirs(base_dir, exist_ok=True)
                os.makedirs(os.path.join(base_dir, "Etc"), exist_ok=True)
                utc_path = os.path.join(base_dir, "UTC")
                etc_utc_path = os.path.join(base_dir, "Etc", "UTC")
                # Remove existing files first (they may have wrong format)
                for path in [utc_path, etc_utc_path]:
                    if os.path.exists(path):
                        os.remove(path)
                with open(utc_path, 'wb') as f:
                    f.write(tzif_data)
                with open(etc_utc_path, 'wb') as f:
                    f.write(tzif_data)
                return utc_path, etc_utc_path
            except Exception as e:
                return None, str(e)
        
        home = os.environ.get("HOME", tempfile.gettempdir())
        created_paths = []
        
        # 1. Create in user's local share (for TZDIR)
        user_tzdir = os.path.join(home, ".local", "share", "zoneinfo")
        result = write_tz_files(user_tzdir)
        if result[0]:
            created_paths.append(user_tzdir)
            os.environ["TZDIR"] = user_tzdir
        
        # 2. Create in pg0's data directory if it exists
        pg0_dir = os.path.join(home, ".pg0")
        if os.path.exists(pg0_dir):
            # Find PostgreSQL share directories within pg0
            for share_pattern in [
                os.path.join(pg0_dir, "**", "share", "postgresql"),
                os.path.join(pg0_dir, "**", "share"),
                os.path.join(pg0_dir, "pgsql", "share"),
            ]:
                for share_dir in glob.glob(share_pattern, recursive=True):
                    tz_dir = os.path.join(share_dir, "timezone")
                    result = write_tz_files(tz_dir)
                    if result[0]:
                        created_paths.append(tz_dir)
                    # Also try nested timezone/timezone (Hypothesis N)
                    nested_tz_dir = os.path.join(share_dir, "timezone", "timezone")
                    result = write_tz_files(nested_tz_dir)
                    if result[0]:
                        created_paths.append(nested_tz_dir)
            
            # Also try direct timezone in pg0 dir
            pg0_tz_dir = os.path.join(pg0_dir, "timezone")
            result = write_tz_files(pg0_tz_dir)
            if result[0]:
                created_paths.append(pg0_tz_dir)
        
        # #region agent log
        # Verify the TZif file is actually valid
        tzif_verify = {}
        for path in created_paths:
            utc_file = os.path.join(path, "UTC")
            if os.path.exists(utc_file):
                with open(utc_file, 'rb') as f:
                    content = f.read()
                    tzif_verify[path] = {
                        "size": len(content),
                        "magic": content[:5].hex() if len(content) >= 5 else "too_short",
                        "valid_magic": content[:4] == b'TZif',
                    }
        
        # Check pg0's data directories
        pg0_data_info = {}
        for data_pattern in [
            os.path.join(pg0_dir, "data"),
            os.path.join(pg0_dir, "hindsight"),  # Named instance
            os.path.join(pg0_dir, "instance", "*"),
        ]:
            for data_dir in glob.glob(data_pattern):
                if os.path.isdir(data_dir):
                    pg_conf = os.path.join(data_dir, "postgresql.conf")
                    pg0_data_info[data_dir] = {
                        "exists": True,
                        "has_pg_conf": os.path.exists(pg_conf),
                        "files": os.listdir(data_dir)[:10] if os.path.isdir(data_dir) else [],
                    }
        
        _debug_log("G", "pg0.py:ensure_tz", "Created timezone files", {
            "created_paths": created_paths,
            "TZDIR": os.environ.get("TZDIR"),
            "pg0_dir_exists": os.path.exists(pg0_dir),
            "tzif_verify": tzif_verify,
            "pg0_data_dirs": pg0_data_info,
        })
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
        # Check for and clean up corrupted data directory
        import glob
        import shutil
        home = os.environ.get("HOME", "/tmp")
        pg0_dir = os.path.join(home, ".pg0")
        
        # Hypothesis L: Check if there's an existing data directory that might be corrupted
        data_dirs_to_check = [
            os.path.join(pg0_dir, "data", "hindsight"),
            os.path.join(pg0_dir, "hindsight"),
            os.path.join(pg0_dir, "data"),
        ]
        existing_data_dirs = []
        for d in data_dirs_to_check:
            if os.path.exists(d):
                existing_data_dirs.append(d)
                # Check if it has a postgresql.conf
                pg_conf = os.path.join(d, "postgresql.conf")
                pg_conf_content = None
                if os.path.exists(pg_conf):
                    try:
                        with open(pg_conf, 'r') as f:
                            pg_conf_content = f.read()[:500]
                    except:
                        pg_conf_content = "read_error"
                _debug_log("L", "pg0.py:start:data_dir_check", f"Found data dir: {d}", {
                    "path": d,
                    "has_pg_conf": os.path.exists(pg_conf),
                    "pg_conf_content": pg_conf_content,
                    "files": os.listdir(d)[:15] if os.path.isdir(d) else [],
                })
        
        # Check pg0's installation timezone directory
        pg0_install_tz_dirs = glob.glob(os.path.join(pg0_dir, "installation", "*", "share", "timezone"))
        for tz_dir in pg0_install_tz_dirs:
            utc_file = os.path.join(tz_dir, "UTC")
            # Check for nested timezone directories (Hypothesis N)
            nested_tz = os.path.join(tz_dir, "timezone")
            _debug_log("J", "pg0.py:start:install_tz_check", f"Checking pg0 install timezone dir", {
                "tz_dir": tz_dir,
                "utc_exists": os.path.exists(utc_file),
                "tz_dir_contents": os.listdir(tz_dir)[:20] if os.path.isdir(tz_dir) else [],
                "nested_tz_exists": os.path.isdir(nested_tz),
                "nested_tz_contents": os.listdir(nested_tz)[:10] if os.path.isdir(nested_tz) else [],
            })
            if os.path.exists(utc_file):
                with open(utc_file, 'rb') as f:
                    content = f.read()
                    _debug_log("P", "pg0.py:start:utc_file_content", f"FULL UTC file content", {
                        "path": utc_file,
                        "size": len(content),
                        "full_hex": content.hex(),  # Full file content
                        "valid_tzif": content[:4] == b'TZif',
                    })
        
        # Hypothesis N: Check for pg_config to understand share directory
        pg_config_paths = glob.glob(os.path.join(pg0_dir, "installation", "*", "bin", "pg_config"))
        for pg_config in pg_config_paths:
            import subprocess
            try:
                sharedir_result = subprocess.run([pg_config, "--sharedir"], capture_output=True, text=True, timeout=5)
                _debug_log("N", "pg0.py:start:pg_config", "pg_config --sharedir result", {
                    "pg_config_path": pg_config,
                    "sharedir": sharedir_result.stdout.strip(),
                    "stderr": sharedir_result.stderr.strip(),
                    "returncode": sharedir_result.returncode,
                })
            except Exception as e:
                _debug_log("N", "pg0.py:start:pg_config_error", f"pg_config failed: {e}", {"path": pg_config})
        
        # Check if there's a postgresql subdirectory in share
        pg0_share_dirs = glob.glob(os.path.join(pg0_dir, "installation", "*", "share"))
        for share_dir in pg0_share_dirs:
            _debug_log("N", "pg0.py:start:share_dir_contents", "pg0 share directory contents", {
                "share_dir": share_dir,
                "contents": os.listdir(share_dir)[:30] if os.path.isdir(share_dir) else [],
                "postgresql_subdir_exists": os.path.isdir(os.path.join(share_dir, "postgresql")),
                "postgresql_tz_exists": os.path.isdir(os.path.join(share_dir, "postgresql", "timezone")),
            })
        # #endregion
        
        # #region agent log
        # Try to find pg0's installation directory
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
                # Try to get pg0 logs for more details
                try:
                    pg0_logs = pg0.logs()
                    pg0_logs_str = str(pg0_logs)[:1000] if pg0_logs else "None"
                except:
                    pg0_logs_str = "Could not get logs"
                _debug_log("C", "pg0.py:start:exception", f"pg0 start failed attempt {attempt}", {
                    "error": last_error, 
                    "error_type": type(e).__name__, 
                    "attempt": attempt,
                    "pg0_logs": pg0_logs_str
                })
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
