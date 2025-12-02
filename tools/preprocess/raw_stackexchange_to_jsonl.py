# Part of the code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange
# --------------------------------------------------------
#
# This tool is used for converting the raw Stack Exchange data downloaded from
# from Archive (ref: https://archive.org/download/stackexchange) to several
# jsonl files.
#
# For downloading process, please refer to:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange
#
# Notice: before you downloading, converting or processing, you might make sure
# that your drive space is large enough to store the raw data (over 100GB),
# converted data (over 100GB)

import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import fire
from loguru import logger
from tqdm import tqdm


@logger.catch(reraise=True)
def get_sites_count(path, topk=28):
    """
    Take top-K sites(`.xml`) by its size of content
    :param path: path to stack_exchage data
    :param topk: number of top-k sites
    :return
        1) a dict stores pair of site and its size of content
        2) a list of topk sites
    """

    logger.info("Got counts for all sites.")
    sites = os.listdir(path)
    sites = [x for x in sites if x.endswith(".xml")]
    counts = {}
    for site in tqdm(sites):
        if site == ".DS_Store":
            continue
        # read xml file and count contents
        with open(os.path.join(path, site), "r") as f:
            # read # lines
            count = sum(1 for line in f)
            counts[site] = count - 3  # subtract the header
    # sort the counts
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    # take first 28
    sites = list(counts.keys())[:topk]
    return counts, sites


@logger.catch(reraise=True)
def get_parents(site, counts):
    """
    Find all answers's parent id, and groups by parent id
    :param site: site(xml) name
    :param counts: a dict stores pair of site and its size of content
    :return: a dict stores pair of parent question id and list of answer id
    """
    parents = {}
    with open(site, "r") as f:
        for i, line in enumerate(tqdm(f, total=counts[os.path.basename(site)])):
            # first 2 lines are header
            # e.g., counts = 2: total=5 lines, 2,3 are data
            # last line is footer
            if i > 1 and i <= counts[os.path.basename(site)] + 1:
                root = ET.fromstring(line)
                if "ParentId" in root.attrib:
                    # this is an answer
                    if root.attrib["ParentId"] not in parents:
                        parents[root.attrib["ParentId"]] = []
                    parents[root.attrib["ParentId"]].append(
                        {"id": root.attrib["Id"], "text": root.attrib["Body"], "score": root.attrib["Score"]}
                    )
    logger.info((f"Got {len(parents)} questions for {site}."))
    return parents


@logger.catch(reraise=True)
def get_qapairs(site, counts, parents):
    """
    Find and group all matched pairs of question and answer in site file
    :param site: site(.xml) name
    :param counts: a dict stores pair of site and its size of content
    :param parents: a dict stores pair of parent question id and
                    list of answer id
    :return: a list of qa pairs
    """
    qa_pairs = []
    with open(site, "r") as f:
        for i, line in enumerate(tqdm(f, total=counts[os.path.basename(site)])):
            # first 2 lines are header
            # e.g., counts = 2: total=5 lines, 2,3 are data
            # last line is footer
            if i > 1 and i <= counts[os.path.basename(site)] + 1:
                root = ET.fromstring(line)
                if "ParentId" not in root.attrib:
                    post_id = root.attrib["Id"]
                    if post_id in parents:
                        # this is a question
                        qa_pairs.append(
                            {
                                "question": {
                                    "id": post_id,
                                    "text": f"{root.attrib['Title']} \
                                  {root.attrib['Body']}",
                                    "score": root.attrib["Score"],
                                },
                                "answers": parents[post_id],
                            }
                        )
                    else:
                        if "Title" in root.attrib:
                            # if there's a title => then a valid question
                            body = root.attrib["Body"] if "Body" in root.attrib else ""
                            score = root.attrib["Score"] if "Score" in root.attrib else 0
                            qa_pairs.append(
                                {
                                    "question": {
                                        "id": post_id,
                                        "text": f"{root.attrib['Title']} {body}",
                                        "score": score,
                                    },
                                }
                            )
    logger.info((f"Got {len(qa_pairs)} qa_pairs for {site}."))
    return qa_pairs


@logger.catch(reraise=True)
def process_qa_pair(pair, site_name, site_count):
    """
    Sort answers by their score for question in qa pair sample,
    add meta info to sample
    :param pair: input qa pair sample
    :param site_name: site name of qa pair
    :param site_count: content size of site
    :return: a dict of qa pair, including ["text", "meta"]
    """
    # sort answers by score
    if "answers" in pair:
        pair["answers"] = sorted(pair["answers"], key=lambda x: x["score"], reverse=True)
        answers = "\nA: ".join([x["text"] for x in pair["answers"]])
        text = f"Q: {pair['question']['text']}\nA: {answers}"
    else:
        text = f"Q: {pair['question']['text']}"
    return {
        "text": text,
        "meta": {
            "site_count": site_count,
            "url": f"https://{site_name}/questions/{pair['question']['id']}",
            "timestamp": "2023-03-29",
            "source": "stackexchange",
            "question_score": pair["question"]["score"],
        },
    }


@logger.catch(reraise=True)
def process_site(site, counts, src_dir, target_dir, num_proc=24):
    """
    Convert one raw Stack Exchange site data to jsonl file.
        1) find all answers's parent id and groups by parent id
        2) find matched pair of question and answers
        3) sort  answers by their score for each question
    :param site: site name endwith `".xml"`
    :param counts: dict stores pair of site name and its size
    :param src_dir: dir path of site
    :param target_dir: path to save jsonl file
    :param num_proc: number of process workers. Default it's 24.
    """
    logger.info(f"Processing {site}...")
    logger.info(f"|{site}|{counts[site]}|")
    site_path = os.path.join(src_dir, site)
    parents = get_parents(site_path, counts)
    qa_pairs = get_qapairs(site_path, counts, parents)

    site_name = site.removesuffix(".xml")
    if "stackoverflow_part" in site_name:
        site_name = "stackoverflow.com"

    site_name_list = [site_name] * len(qa_pairs)
    counts_list = [counts[site]] * len(qa_pairs)
    tasks = [*zip(qa_pairs, site_name_list, counts_list)]
    with Pool(num_proc) as p:
        results = p.starmap(process_qa_pair, iterable=tasks)
    logger.info(
        f"Writing {len(results)} results to \
        {os.path.join(target_dir, site_name+'.jsonl')}"
    )

    with open(os.path.join(target_dir, site_name + ".jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


@logger.catch(reraise=True)
def main(src_dir, target_dir, topk=28, num_proc=1):
    """
    Convert the raw Stack Exchange data downloaded from from Archive
    (ref: https://archive.org/download/stackexchange) to several
    jsonl files.
    :param src_dir: if you download raw Stack Exchange data as Redpajama did,
               you will get a directory src which includes hundreds of 7z files
               whose filenames are like "*.*.com.7z ". You need to unzip these
               files and rename the POSTs.xml to the corresponding compressed
               package name and place it in that dir.
    :param target_dir: result directory to store the converted jsonl files.
    :param topk: select the topk sites with the most content.
                  Default it's 28.
    :param num_proc: number of process workers. Default it's 1.
    """
    # check if the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError("The raw stack_exchange source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # select topk sites from src dir by most contents number
    counts, sites = get_sites_count(src_dir, topk=topk)
    for site in sites:
        logger.info(f"Start to process {site}")
        process_site(site, counts, src_dir, target_dir, num_proc)


if __name__ == "__main__":
    fire.Fire(main)
