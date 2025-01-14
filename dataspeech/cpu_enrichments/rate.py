from phonemizer import phonemize
from phonemizer.separator import Separator


def phonemize_vi(text):
    return phonemize(
        text,
        language="vi",
        backend="espeak",
        separator=Separator(phone=None, word=" ", syllable="|"),
        strip=True,
        preserve_punctuation=True,
        njobs=4,
        language_switch="remove-flags",
    )


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(
                batch[text_column_name], batch["speech_duration"]
            ):
                phonemes = phonemize_vi(text)
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = phonemize_vi(text)

                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate

                speaking_rate = len(phonemes) / audio_length

                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = phonemize_vi(batch[text_column_name])
        if "speech_duration" in batch:
            audio_length = (
                batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
            )
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = (
                len(batch[audio_column_name]["array"].squeeze()) / sample_rate
            )

        speaking_rate = len(phonemes) / audio_length

        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch
