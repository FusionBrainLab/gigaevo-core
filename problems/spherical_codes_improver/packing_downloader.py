import requests


def download_packing(
    ndim: int, npoints: int, out_folder="problems/spherical_codes/known_packings"
) -> str:
    url = f"https://spherical-codes.org/data/{ndim}/{npoints}"
    out_path = f"{out_folder}/packing_{ndim}_{npoints}.txt"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    return out_path
