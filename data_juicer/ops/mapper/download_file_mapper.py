import asyncio
import copy
import os
import os.path as osp
from typing import List, Union
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from data_juicer.utils.file_utils import download_file, is_remote_path

from ..base_op import OPERATORS, Mapper

OP_NAME = "download_file_mapper"


@OPERATORS.register_module(OP_NAME)
class DownloadFileMapper(Mapper):
    """Mapper to download URL files to local files or load them into memory.

    This operator downloads files from URLs and can either save them to a specified
    directory or load the contents directly into memory. It supports downloading multiple
    files concurrently and can resume downloads if the `resume_download` flag is set. The
    operator processes nested lists of URLs, flattening them for batch processing and then
    reconstructing the original structure in the output. If both `save_dir` and `save_field`
    are not specified, it defaults to saving the content under the key `image_bytes`. The
    operator logs any failed download attempts and provides error messages for
    troubleshooting."""

    _batched_op = True

    def __init__(
        self,
        download_field: str = None,
        save_dir: str = None,
        save_field: str = None,
        resume_download: bool = False,
        timeout: int = 30,
        max_concurrent: int = 10,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param save_dir: The directory to save downloaded files.
        :param download_field: The filed name to get the url to download.
        :param save_field: The filed name to save the downloaded file content.
        :param resume_download: Whether to resume download. if True, skip the sample if it exists.
        :param timeout: Timeout for download.
        :param max_concurrent: Maximum concurrent downloads.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.download_field = download_field
        self.save_dir = save_dir
        self.save_field = save_field
        self.resume_download = resume_download
        if not (self.save_dir or self.save_field):
            logger.warning(
                "Both `save_dir` and `save_field` are not specified. Use the default `image_bytes` key to "
                "save the downloaded contents."
            )
            self.save_field = self.image_bytes_key
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def download_files_async(self, urls, return_contents, save_dir=None, **kwargs):

        async def _download_file(
            session: aiohttp.ClientSession,
            semaphore: asyncio.Semaphore,
            idx: int,
            url: str,
            save_dir=None,
            return_content=False,
            **kwargs,
        ) -> dict:
            try:
                status, response, content, save_path = "success", None, None, None

                # local file
                if not is_remote_path(url):
                    if return_content:
                        with open(url, "rb") as f:
                            content = f.read()
                    if save_dir:
                        save_path = url
                    return idx, save_path, status, response, content

                # skip already downloaded files
                if not save_dir and not return_content:
                    return idx, save_path, status, response, content

                if save_dir:
                    filename = os.path.basename(urlparse(url).path)
                    save_path = osp.join(save_dir, filename)
                    if os.path.exists(save_path):
                        if return_content:
                            with open(save_path, "rb") as f:
                                content = f.read()
                        return idx, save_path, status, response, content

                async with semaphore:
                    response, content = await download_file(
                        session, url, save_path, return_content=True, timeout=self.timeout, **kwargs
                    )
            except Exception as e:
                status = "failed"
                response = str(e)
                save_path = None
                content = None

            return idx, save_path, status, response, content

        async def run_downloads(urls, return_contents, save_dir=None, **kwargs):
            semaphore = asyncio.Semaphore(self.max_concurrent)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _download_file(session, semaphore, idx, url, save_dir, return_contents[idx], **kwargs)
                    for idx, url in enumerate(urls)
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_downloads(urls, return_contents, save_dir, **kwargs))
        results.sort(key=lambda x: x[0])

        return results

    def _flat_urls(self, nested_urls):
        flat_urls = []
        structure_info = []  # save as original index, sub index

        for idx, urls in enumerate(nested_urls):
            if isinstance(urls, list):
                for sub_idx, url in enumerate(urls):
                    flat_urls.append(url)
                    structure_info.append((idx, sub_idx))
            else:
                flat_urls.append(urls)
                structure_info.append((idx, -1))  # -1 means single str element

        return flat_urls, structure_info

    def _create_path_struct(self, nested_urls, keep_failed_url=True) -> str:
        if keep_failed_url:
            reconstructed = copy.deepcopy(nested_urls)
        else:
            reconstructed = []
            for item in nested_urls:
                if isinstance(item, list):
                    reconstructed.append([None] * len(item))
                else:
                    reconstructed.append(None)

        return reconstructed

    def _create_save_field_struct(self, nested_urls, save_field_contents=None) -> str:
        if save_field_contents is None:
            save_field_contents = []
            for item in nested_urls:
                if isinstance(item, list):
                    save_field_contents.append([None] * len(item))
                else:
                    save_field_contents.append(None)
        else:
            # check whether the save_field_contents format is correct and correct it automatically
            for i, item in enumerate(nested_urls):
                if isinstance(item, list):
                    if not save_field_contents[i] or len(save_field_contents[i]) != len(item):
                        save_field_contents[i] = [None] * len(item)

        return save_field_contents

    def download_nested_urls(self, nested_urls: List[Union[str, List[str]]], save_dir=None, save_field_contents=None):
        flat_urls, structure_info = self._flat_urls(nested_urls)

        if save_field_contents is None:
            # not save contents, set return_contents to False
            return_contents = [False] * len(flat_urls)
        else:
            # if original content None, set bool value to True to get content else False to skip reload it
            return_contents = [not c for sublist in save_field_contents for c in sublist]

        download_results = self.download_files_async(
            flat_urls,
            return_contents,
            save_dir,
        )

        if self.save_dir:
            reconstructed_path = self._create_path_struct(nested_urls)
        else:
            reconstructed_path = None

        failed_info = ""
        for i, (idx, save_path, status, response, content) in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            if status != "success":
                save_path = flat_urls[i]
                failed_info += "\n" + str(response)

            if save_field_contents is not None:
                if return_contents[i]:
                    if sub_idx == -1:
                        save_field_contents[orig_idx] = content
                    else:
                        save_field_contents[orig_idx][sub_idx] = content

            if self.save_dir:
                # TODO: add download stats
                if sub_idx == -1:
                    reconstructed_path[orig_idx] = save_path
                else:
                    reconstructed_path[orig_idx][sub_idx] = save_path

        return save_field_contents, reconstructed_path, failed_info

    def process_batched(self, samples):
        if self.download_field not in samples or not samples[self.download_field]:
            return samples

        batch_nested_urls = samples[self.download_field]

        if self.save_field:
            if not self.resume_download:
                if self.save_field in samples:
                    raise ValueError(
                        f"{self.save_field} is already in samples. '\
                        'If you want to resume download, please set `resume_download=True`"
                    )
                save_field_contents = self._create_save_field_struct(batch_nested_urls)
            else:
                if self.save_field not in samples:
                    save_field_contents = self._create_save_field_struct(batch_nested_urls)
                else:
                    save_field_contents = self._create_save_field_struct(batch_nested_urls, samples[self.save_field])
        else:
            save_field_contents = None

        save_field_contents, reconstructed_path, failed_info = self.download_nested_urls(
            batch_nested_urls, save_dir=self.save_dir, save_field_contents=save_field_contents
        )

        if self.save_dir:
            samples[self.download_field] = reconstructed_path

        if self.save_field:
            samples[self.save_field] = save_field_contents

        if len(failed_info):
            logger.error(f"Failed files:\n{failed_info}")

        return samples
