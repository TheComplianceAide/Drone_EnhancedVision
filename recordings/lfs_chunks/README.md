# Chunked recording recovery

GitHub LFS rejects individual objects over 2 GB. The two larger raw recordings are therefore stored as lossless LFS chunks in this directory.

After cloning with Git LFS installed, restore the original `.mov` files with:

```bash
python3 tools/reconstruct_lfs_recordings.py
```

The tool verifies every chunk and the rebuilt recording against `manifest.json` before replacing an output file.
