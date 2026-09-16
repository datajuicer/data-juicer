# Multimodal Tools

This folder contains some scripts and tools for multimodal datasets before and after using Data-Juicer.

## Convert Absolute Path to Relative Path

After being processed by Data-Juicer, the output multimodal data paths are absolute and could originate from diverse sources, being stored under various root directories.
To convert absolute paths to relative paths and facilitate data migration, we provide a tool to carry out this transformation. This tool also supports copying related multimodal data from multiple sources into a single directory.
Both input and output of this utility conform to Data-Juicer's data format. If you wish to convert the data into another format while maintaining relative paths, please run this tool before proceeding with the dataset format conversion in the following section.
To learn more about the usage of the absolute to relative path conversion tool, you can execute the following command:

```shell
python tools/fmt_conversion/multimodal/absolute_path_to_relative_path.py --help
```

## Dataset Format Conversion

Due to large format diversity among different multimodal datasets and works,
Data-Juicer propose a novel intermediate text-based interleaved data format for multimodal dataset, which
is based on chunk-wise formats such MMC4 dataset.

In the Data-Juicer format, a multimodal sample or document is based on a text,
which consists of several text chunks. Each chunk is a semantic unit, and all the
multimodal information in a chunk should talk about the same thing and be aligned
with each other.

Here is a multimodal sample example in Data-Juicer format below.
- It includes 4 chunks split by the special token `<|__dj__eoc|>`.
- In addition to texts, there are 3 other modalities: images, audios, videos.
They are stored on the disk and their paths are
listed in the corresponding first-level fields in the sample.
- Other modalities are represented as special tokens in the text (e.g. image -- `<__dj__image>`).
The special tokens of each modality correspond to the paths in the order of appearance.
(e.g. the two image tokens in the third chunk are images of antarctica_map and europe_map respectively)
- There could be multiple types of modalities and multiple modality special tokens in a single chunk,
and they are semantically aligned with each other and text in this chunk.
The position of special tokens can be random in a chunk. (In general, they are usually before or after the text.)
- For multimodal samples, unlike text-only samples, the computed stats for other
modalities could be a list of stats for the list of multimodal data (e.g. image_widths in this sample).

```python
{
  "text": "<__dj__image> Antarctica is Earth's southernmost and least-populated continent. <|__dj__eoc|> "
          "<__dj__video> <__dj__audio> Situated almost entirely south of the Antarctic Circle and surrounded by the "
          "Southern Ocean (also known as the Antarctic Ocean), it contains the geographic South Pole. <|__dj__eoc|> "
          "Antarctica is the fifth-largest continent, being about 40% larger than Europe, "
          "and has an area of 14,200,000 km2 (5,500,000 sq mi). <__dj__image> <__dj__image> <|__dj__eoc|> "
          "Most of Antarctica is covered by the Antarctic ice sheet, "
          "with an average thickness of 1.9 km (1.2 mi). <|__dj__eoc|>",
  "images": [
    "path/to/the/image/of/antarctica_snowfield",
    "path/to/the/image/of/antarctica_map",
    "path/to/the/image/of/europe_map"
  ],
  "audios": [
    "path/to/the/audio/of/sound_of_waves_in_Antarctic_Ocean"
  ],
  "videos": [
    "path/to/the/video/of/remote_sensing_view_of_antarctica"
  ],
  "meta": {
    "src": "customized",
    "version": "0.1",
    "author": "xxx"
  },
  "stats": {
    "lang": "en",
    "image_widths": [224, 336, 512],
    ...
  }
}
```

According to this format, Data-Juicer provided several dataset format conversion tools for some popular multimodal
works.

These tools consist of two types:
- Other format to Data-Juicer format: These tools are in `source_format_to_data_juicer_format` directory. They help to convert datasets in other formats to target datasets in Data-Juicer format.
- Data-Juicer format to other format: These tools are in `data_juicer_format_to_target_format` directory. They help to convert datasets in Data-Juicer formats to target datasets in target format.

For now, dataset formats that are supported by Data-Juicer are listed in the following table.

| Format             | Type       | source_format_to_data_juicer_format | data_juicer_format_to_target_format | Ref.                                                                                                             |
|--------------------|------------|-------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------|
| LLaVA-like         | image-text | `llava_to_dj.py`                    | `dj_to_llava.py`                    | [Format Description](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format) |
| MMC4-like          | image-text | `mmc4_to_dj.py`                     | `dj_to_mmc4.py`                     | [Format Description](https://github.com/allenai/mmc4#documents)                                                  |
| WavCaps-like       | audio-text | `wavcaps_to_dj.py`                  | `dj_to_wavcaps.py`                  | [Format Description](https://github.com/XinhaoMei/WavCaps#table-of-contents)                                     |
| Video-ChatGPT-like | video-text | `video_chatgpt_to_dj.py`            | `dj_to_video_chatgpt.py`                | [Format Description]( https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data)                                                                                           |                                                                                          |
| Youku-mPLUG-like   | video-text | `youku_to_dj.py`                    | `dj_to_youku.py`                    | [Format Description](https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary)                          |                                                                                          |
| InternVid-like     | video-text | `internvid_to_dj.py`                | `dj_to_internvid.py`                | [Format Description](https://huggingface.co/datasets/OpenGVLab/InternVid)                                        |                                                                                          |


For all tools, you can run the following command to find out the usage of them:

```shell
# e.g. llava_to_dj.py
python tools/fmt_conversion/multimodal/source_format_to_data_juicer_format/llava_to_dj.py --help
```

Before using these tools, you might need to take a glance at the reference
materials in the above tables for each format, to better know the detail format
information and understand the arguments for each tool.

### Notice
There might be some tiny differences after converting a source dataset to Data-Juicer
format and convert it back. However, these differences have nearly no effects
on the semantics of datasets. Here we will show these tiny differences in detail
for each source format.

#### LLaVA-like
The format of LLaVA-like datasets are defined [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format).
Although it's simple, but in real scenarios, there might be some slight variations
in some samples.

Here we take the [visual instruction tuning dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) as an example,
and show how these variations influence the dataset format. The table below
shows the number of different samples between the original dataset and the
dataset after processing. There are 665,298 samples in the original dataset.

| process                                                                                | # of diff.  |
|----------------------------------------------------------------------------------------|-------------|
| 1. apply `llava_to_dj.py` and `dj_to_llava.py`                                         | 113,501     |
| 2. convert integer ids to string ids in the original dataset                           | 41,361      |
| 3. strip whitespaces before and after values of conversations in the original dataset  | 40,688      |
| 4. add `'model': ''` fields in the converted dataset                                   | 1           |

It's worth noticing that processes 2-4 won't influence the semantics of sample conversations in the dataset.
Thus we think the dataset after conversion can align with the original dataset.

Finally, the only 1 sample is different because there are some extra useless fields ("text", "markdown")
in the conversations, which is shown below. But the "from" and "value" fields are the same between original
and converted datasets, so we can regard this sample is aligned with the original one as well.

```json
# original conversations
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "text":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\nSlow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\nServer overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\nLarge dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\nAPI endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\nCheck your network connection and ensure that it is stable.\n\nVerify that the server hosting the API is not overloaded with requests.\n\nOptimize the query to retrieve only the necessary data.\n\nTry using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "markdown":{
            "type":"answer-markdown",
            "index":1,
            "answer":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n\n2. Verify that the server hosting the API is not overloaded with requests.\n\n3. Optimize the query to retrieve only the necessary data.\n\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
        }
    }
]

# converted conversations
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
    }
]
```

#### MMC4-like

The format of MMC4-like datasets are defined [here](https://github.com/allenai/mmc4#documents). Except `image_info` and `text_list`,
which are used when converting them to Data-Juicer format, there is an important field `similarity_matrix`. Similarity matrix is
a matrix of shape `len(image_info) x len(text_list)`, which means it highly depends on the numbers of images and text sentences and their
orders.

However, when processing such datasets with Data-Juicer, images or sentences might be removed from a sample by Filters, and they could be
modified by some Mappers. Thus, after processing, this similarity matrix might be no longer aligned with `image_info` or `text_list`.
Users should be cautious about this point if you need this matrix in later usages.

Despite these extra fields, tools for MMC4 can perfectly convert MMC4-like datasets to Data-Juicer-format datasets and convert them back~

#### WavCaps-like

The [WavCaps](https://github.com/XinhaoMei/WavCaps#dataset) is composed of four sub-datasets: [FreeSound](https://freesound.org/), [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/),[SoundBible](https://soundbible.com/) and [AudioSet Strongly-labelled Subset](https://research.google.com/audioset/download_strong.html). Each sub-dataset has different fields. For example, the 'description' field is included in SoundBible, but does not exist in AudioSet. To ensure that the different sub-datasets can be properly merged after conversion, the union of all fields from the sub-datasets is used during the wavcaps_to_dj stage, and all fields are fully retained during the dj_to_wavcaps stage.

```json
# original dataset
{ "num_captions_per_audio": 1,
  "data": [{
        "title": "Airplane Landing Airport",
        "description": "Large commercial airplane landing at an airport runway.",
        "author": "Daniel Simion",
        "href": "2219-Airplane-Landing-Airport.html",
        "caption": "An airplane is landing.",
        "id": "2219",
        "duration": 14.1424375,
        "audio": "wav_path",
        "download_link": "http://soundbible.com/grab.php?id=2219&type=wav"}]
}

# converted dataset
{ "num_captions_per_audio": 1,
  "data": [{
        "title": "Airplane Landing Airport",
        "description": "Large commercial airplane landing at an airport runway.",
        "author": "Daniel Simion",
        "href": "2219-Airplane-Landing-Airport.html",
        "caption": "An airplane is landing.",
        "id": "2219",
        "duration": 14.1424375,
        "audio": "wav_path",
        "download_link": "http://soundbible.com/grab.php?id=2219&type=wav",
        "category": "",
        "tags": "" }]
}
```

#### Video-ChatGPT-like

The Video-ChatGPT dataset contains 3 types of data with unified format:
- Topics for Video summarization
- Description-based question-answers (exploring spatial, temporal, relationships, and reasoning concepts);
- and Creative/generative question-answers.
They all obey the `<question, answer, video_id>` format, where the `video_id` is in the form "v_youtube_id". We suppose that users have downloaded these videos already, and they need to specify the corresponding storage directory when using the converter tool.



#### Youku-mPLUG-like

The Youku-mPLUG dataset contains 4 types of format: pretrain, classification, retrieval, captioning.
They are slightly different from each other in field name or other attributes, but all of them obey the `<video, caption>` format.

#### InternVid-like

The InternVid dataset contains 4 fields:
- `YoutubeID`: the Youtube ID of the video used in the sample.
We suppose that users have downloaded these videos already
and this field is replaced with its storage path.
- `Start_timestamp`: the start timestamp in string of the video clip for the
corresponding caption.
- `End_timestamp`: the end timestamp in string of the video clip for the
corresponding caption.
- `Caption`: the corresponding caption for the video clip.

As we can see, the caption in this dataset corresponds to the video clip
specified by the start/end timestamps instead of the whole video. So the
conversion tool will cut the specified video clip for you if the argument
`cut_videos` is set to True. You can cut before conversion by yourself as well.

#### MSR-VTT-like
MSR-VTT dataset contains multiple fields, here we use 2 fields:
- `video_id`: the video file name without suffix used in the sample.
We suppose that users have downloaded these videos already。
- `caption`: the corresponding caption for the video.
