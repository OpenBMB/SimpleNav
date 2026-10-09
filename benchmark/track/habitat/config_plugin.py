from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class TrackConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="simplenav-track", path="pkg://benchmark.track.habitat.config")


Plugins.instance().register(TrackConfigPlugin)
