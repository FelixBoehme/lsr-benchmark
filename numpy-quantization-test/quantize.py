from pathlib import Path

import click
import numpy as np


def pack_4bit(arr: np.array):
    if len(arr) % 2 != 0:
        arr = np.append(arr, 0)
    pairs = arr.reshape(-1, 2)
    packed = (pairs[:, 0] << 4) | pairs[:, 1]

    return packed.astype(np.uint8)


def pack_2bit(arr):
    remainder = len(arr) % 4
    if remainder != 0:
        arr = np.append(arr, np.zeros(4 - remainder, dtype=np.uint8))
    groups = arr.reshape(-1, 4)
    packed = (groups[:, 0] << 6) | (groups[:, 1] << 4) | (groups[:, 2] << 2) | (groups[:, 3])

    return packed.astype(np.uint8)


def quantize(embeddings: np.ndarray, level: int, bitpack: bool):
    match level:
        case 1 | 2 | 4:
            normalized = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
            quantized = np.round(normalized * (2**level - 1)).astype(np.int8)
            if not bitpack:
                return quantized
            else:
                if level == 1:
                    return np.packbits(quantized)
                elif level == 2:
                    return pack_2bit(quantized)
                else:
                    return pack_4bit(quantized)
        case 8:
            return (embeddings * 255).astype(np.int8)
        case 16:
            return embeddings.astype(np.float16)
        case _:
            raise ValueError(f"Quantizing to {level} bits is not supported.")


@click.command()
@click.argument("level", type=int, required=True, nargs=-1)
@click.option("--src", type=click.Path(exists=True, readable=True, path_type=Path), required=True)
@click.option("--dest", type=click.Path(exists=True, writable=True, path_type=Path), required=True)
@click.option("--compress", is_flag=True)
@click.option("--bitpack", is_flag=True)
def main(level: list[int], src: Path, dest: Path, compress: bool, bitpack: bool):
    npzFile = np.load(src)
    embeddings = npzFile["data"]
    save = np.savez_compressed if compress else np.savez
    for l in level:
        out_path = dest / Path("-".join([src.stem, str(l), "bit", "compressed" if compress else "uncompressed", "bitpacked" if bitpack else "unpacked"]))
        quantized_embeddings = quantize(embeddings, l, bitpack)
        save(out_path, data=quantized_embeddings, indices=npzFile["indices"], indptr=npzFile["indptr"])


if __name__ == "__main__":
    main()
