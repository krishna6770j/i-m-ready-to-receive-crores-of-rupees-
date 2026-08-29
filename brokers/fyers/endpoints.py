"""Verified FYERS API facts, with the provenance of each one.

Manager correction #8 requires that every endpoint and parameter be traceable to
a source. Nothing in this file was guessed.

PRIMARY SOURCE -- installed SDK ``fyers-apiv3==3.1.16``, module
``fyers_apiv3/fyersModel.py``, class ``Config`` and the methods named below.
Read directly from site-packages on 2026-08-29.

SECONDARY SOURCE -- FYERS Data API knowledge base:
https://support.fyers.in/portal/en/kb/fyers-api-integrations/fyers-api/api-v3/data-api

NOT VERIFIED -- the JSON response body shape of /history. The formal reference
at myapi.fyers.in/docsv3 is a JavaScript-rendered SPA that could not be
retrieved, and the domain is blocked to the browser tool. The response parser in
``marketdata.schemas.from_fyers_candles`` therefore validates the payload shape
defensively and raises a precise error on mismatch instead of assuming.
"""

from __future__ import annotations

# --- Base URLs -----------------------------------------------------------
# Source: fyersModel.Config.API / Config.DATA_API (SDK source, verified).
API_BASE = "https://api-t1.fyers.in/api/v3"
DATA_API_BASE = "https://api-t1.fyers.in/data"

# --- Endpoint paths ------------------------------------------------------
# Source: fyersModel.Config attributes (SDK source, verified).
AUTH_PATH = "/generate-authcode"          # Config.auth            -> API_BASE
VALIDATE_AUTHCODE_PATH = "/validate-authcode"  # Config.generate_access_token -> API_BASE
HISTORY_PATH = "/history"                 # Config.history         -> DATA_API_BASE
QUOTES_PATH = "/quotes"                   # Config.quotes          -> DATA_API_BASE
PROFILE_PATH = "/profile"                 # Config.get_profile     -> API_BASE

# Full URL for historical candles. Source: FyersModel.history calls
# service.get_call(Config.history, ..., data_flag=True), and get_call builds
# `Config.DATA_API + api` when data_flag is True (SDK source, verified).
HISTORY_URL = f"{DATA_API_BASE}{HISTORY_PATH}"

# --- Request headers -----------------------------------------------------
# Source: FyersServiceSync.get_call (SDK source, verified). The Authorization
# value is "{client_id}:{access_token}".
HEADER_VERSION = "3"


def authorization_header(client_id: str, access_token: str) -> str:
    """Build the Authorization header value.

    Source: ``FyersModel.__init__`` sets
    ``self.header = "{}:{}".format(self.client_id, self.token)``.
    """
    return f"{client_id}:{access_token}"


# --- /history parameters -------------------------------------------------
# Source: FyersModel.history docstring (SDK source, verified verbatim):
#   symbol (str): Symbol of the product. Eg: 'NSE:SBIN-EQ'.
#   resolution (str): 'Day' or '1D', '1', '2', '3', '5', '10', '15', '20',
#                     '30', '60', '120', '240'.
#   date_format (int): 0 to enter the epoch value, 1 for 'yyyy-mm-dd'.
#   range_from (str): Start date.
#   range_to (str): End date.
#   cont_flag (int): 1 for continuous data.
HISTORY_PARAMS = (
    "symbol",
    "resolution",
    "date_format",
    "range_from",
    "range_to",
    "cont_flag",
)

DATE_FORMAT_EPOCH = 0
DATE_FORMAT_YMD = 1

# --- Per-request range limit ---------------------------------------------
# NOT VERIFIED from official documentation. FYERS community posts state 100
# days per request for minute resolutions and 366 for daily. The SDK does not
# encode any limit. This value is used as a CONSERVATIVE chunk size, and the
# downloader reports what it actually received versus requested so a wrong
# assumption becomes visible in the data rather than silently truncating.
ASSUMED_MAX_DAYS_PER_REQUEST_INTRADAY = 100
ASSUMED_MAX_DAYS_PER_REQUEST_DAILY = 366

# --- Rate limits ---------------------------------------------------------
# Per-day figure is official (https://fyers.in/products/api states
# "Upto 1 Lakh requests per day"). Per-second and per-minute figures come from
# FYERS community posts and are NOT officially confirmed; they are used as
# conservative self-imposed throttles.
RATE_LIMIT_PER_DAY = 100_000          # official
ASSUMED_RATE_LIMIT_PER_SECOND = 10    # unverified, treated as a ceiling
ASSUMED_RATE_LIMIT_PER_MINUTE = 200   # unverified, treated as a ceiling

# --- Response envelope ---------------------------------------------------
# The FYERS response envelope uses "s" for status with the value "ok" on
# success. Observed in SDK error handling and community examples; the exact
# error-code vocabulary is NOT verified, so the adapter treats any non-"ok"
# status as an error and surfaces the raw payload.
STATUS_KEY = "s"
STATUS_OK = "ok"
CANDLES_KEY = "candles"
MESSAGE_KEY = "message"
