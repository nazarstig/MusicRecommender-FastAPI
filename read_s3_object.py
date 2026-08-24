"""
Read or list objects in the S3/MinIO bucket this project uses (matrices,
evaluation results, etc.) via the S3 API - MinIO stores object bytes inside
its own xl.meta/part.N format on disk, so poking at the data directory
directly won't show you anything readable; this goes through boto3 instead.

Usage:
    python read_s3_object.py --list                        # list every key in the bucket
    python read_s3_object.py --list evaluations/            # list keys under a prefix
    python read_s3_object.py evaluations/k_search_20260824_130124.csv                       # print text content
    python read_s3_object.py models/latest/recommendations_matrices.npz --out matrices.npz  # save binary content
"""
import argparse

from app.services.s3_service import S3Service


def main():
    parser = argparse.ArgumentParser(description="Read or list objects in the project's S3/MinIO bucket.")
    parser.add_argument("key", nargs="?", help="Object key to read (omit when using --list)")
    parser.add_argument(
        "--list", nargs="?", const="", default=None, metavar="PREFIX",
        help="List keys instead of reading one, optionally filtered by prefix",
    )
    parser.add_argument(
        "--out", help="Save the object to this local path instead of printing it (needed for binary files like .npz)"
    )
    args = parser.parse_args()

    s3 = S3Service()

    if args.list is not None:
        keys = s3.list_keys(args.list)
        if not keys:
            print("(no objects found)")
        for key, size, last_modified in keys:
            print(f"{last_modified}  {size:>10} bytes  {key}")
        return

    if not args.key:
        parser.error("provide a key to read, or use --list")

    data = s3.download_bytes(args.key)

    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"Saved {len(data)} bytes to {args.out}")
    else:
        try:
            print(data.decode("utf-8"))
        except UnicodeDecodeError:
            print(f"Object is {len(data)} bytes of binary data - use --out <path> to save it instead of printing.")


if __name__ == "__main__":
    main()
