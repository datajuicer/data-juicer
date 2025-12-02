# Most of the code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

import string

import emoji

# special characters
MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits + string.whitespace
OTHER_SPECIAL_CHARACTERS = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(MAIN_SPECIAL_CHARACTERS + OTHER_SPECIAL_CHARACTERS)
SPECIAL_CHARACTERS.update(EMOJI)

# various whitespaces for whitespace normalization
# whitespaces in unicode can be found here:
# https://en.wikipedia.org/wiki/Whitespace_character
VARIOUS_WHITESPACES = {
    " ",
    "	",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
    "​",
    "‌",
    "‍",
    "⁠",
    "￼",
    "",
}
