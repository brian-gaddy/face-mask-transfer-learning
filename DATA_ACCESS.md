# Data Access

The full face-mask image archive is intentionally excluded from this repository.

The source archive contains more than 8,000 images and is approximately 318 MB, so the
binary dataset is kept outside normal Git history. This keeps the repository lightweight
and avoids distributing a large image corpus as source code.

## Expected local layout

After obtaining the dataset, place the class images under `data/raw/` and use the notebook
to create training, validation, and test splits.

The source classes are:

- `with_mask`
- `without_mask`
- `incorrect_mask`

Dataset directories are ignored by Git. Do not commit the full image archive or generated
train/validation/test folders.
