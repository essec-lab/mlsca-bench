# Known issues in the published datasets

Problems found in the data as published by its authors, from loading and checking the real files in September and October 2026. MLSCA-Bench downloads the files unchanged; each issue is also noted in the dataset's registry entry (`mlsca-bench info NAME`).

## eshard-aes-masked-shuffled: not a separate shuffled acquisition

`Nucleo_AES_masked_shuffled.ets` was compared with `Nucleo_AES_masked_non_shuffled.ets` from the same repository:

- all 100,000 traces are byte-for-byte identical in the two files;
- the keys, masks and ciphertexts are identical too;
- only the plaintexts differ, and the shuffled file adds a `permutation` column;
- AES-128(plaintext, key) equals the ciphertext for every trace of the non-shuffled file and for none of the shuffled file.

The shuffled file therefore seems to reuse the non-shuffled acquisition with a different plaintext column. Part of this is reported upstream: [issue #2](https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled/-/issues/2) (open since October 2023) says the ciphertexts are wrong. **Do not use it as a separate shuffled dataset** until the authors correct it; `eshard-aes-masked-non-shuffled` is consistent.

## ches-ctf-2018-challenge-2: the two no-key files store a zero key

`PinataAcqTask2.5_1k_NK_upload.trs` and `PinataAcqTask2.6_1k_NK_upload.trs` (2,000 of the 42,000 traces) have an all-zero key field: the key was not provided ("NK"). Labels computed from it are wrong, so leave these files out of profiling (`files=["PinataAcqTask2.[1-4]*"]`) or supply the key yourself. Files 2.1 to 2.3 use a new key per trace and 2.4 one fixed key; in all four, AES-128(plaintext, key) equals the ciphertext.

## ascadv-raw and ascadv2r: the last record is empty

The last record of each raw file is all zeros, trace and metadata (plaintext, key, masks): record 300,000 of the 300,001 in `ascadv-raw`, and the last of the 100,000 records in the `ascadv2r` files (checked on the first file). ANSSI documents this for ASCADv2 as an acquisition issue. Drop the last record before use to avoid bias.

## ascadv2: five zeroed profiling traces

Per the ANSSI notice, profiling traces 99999, 199999, 299999, 399999 and 499999 are zeroed by the same acquisition issue; the attack part is clean. Drop them before training.

## dpacontest-v4-2: implementation bug in the published traces

From the [official documentation](https://dpacontest.telecom-paris.fr/v4/42_doc.php): the traces were updated in July 2015 and no longer contain the first bug found (October 2014), but they still contain the second one (August 2015): "the permutation function Shuffle10 is used before the first round instead of Shuffle0". The mask set used also differs from the published description (page 4 of the SPACE 2014 paper): `[0x03, 0x0c, 0x35, 0x3a, 0x50, 0x5f, 0x66, 0x69, 0x96, 0x99, 0xa0, 0xaf, 0xc5, 0xca, 0xf3, 0xfc]`. The authors ask users of these traces to cite *RSM: A small and fast countermeasure for AES, secure against 1st and 2nd-order zero-offset SCAs* (DATE 2012).

## dpacontest-v3-secmatv3-des-20071219: zip over 4 GB without zip64

The joined 6.3 GB archive was written without zip64, so its header offsets overflow (7-Zip: "32-bit overflow in headers"). Python's zip reader and standard `unzip` reject it; MLSCA-Bench extracts it with 7-Zip, which reads all 67,753 files correctly. Nothing is lost, but extracting it needs the `7z` command.

## Reporting a new issue

If a dataset's files do not match its documentation (for example, ciphertexts that are not the encryption of the plaintexts), the problem may be in the published data. Please [open an issue](../CONTRIBUTING.md#reporting-a-broken-link-or-wrong-metadata) with the `verify` output.
