from typing import Final
import slowapi
import slowapi.util


limiter: Final[slowapi.Limiter] = slowapi.Limiter(slowapi.util.get_remote_address)
