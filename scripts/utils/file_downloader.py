from uuid import uuid4
from typing import List


import tator


class FileDownloader:
    def __init__(self, work_dir: str, api: tator.api):
        self._work_dir = work_dir
        self._api = api

    def __call__(self, media_ids: List[int]) -> str:
        """
        Downloads media associated with an id and returns the path to which it was downloaded. If an
        exception occurs during download, an empty path is returned, signifying no file was
        downloaded.
        """
        media_paths = []
        for media_id in media_ids:
            media_element = self._api.get_media(media_id)
            media_path = os.path.join(self._work_dir, str(uuid4()))
            try:
                for _ in tator.util.download_media(api, media_element, image_path):
                    pass
            except:
                media_paths.append("")
            else:
                media_paths.append(media_path)

        return media_paths
