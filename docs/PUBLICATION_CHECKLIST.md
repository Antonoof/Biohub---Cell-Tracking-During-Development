# Public release checklist

Complete this checklist before publishing or merging the release branch.

## Identity and citation

- [x] William Duckworth is the desired displayed author name.
- [ ] `CITATION.cff` contains the final repository URL, release tag, and DOI if
      an archive DOI is issued.
- [x] The Git commit uses the privacy-preserving `tweak36@users.noreply.github.com` address.

## Licensing and attribution

- [x] Apache-2.0 has been selected for William Duckworth's original source.
- [ ] Upstream source files have been identified and retain required notices.
- [ ] Dataset and checkpoint terms permit the identifiers and artifacts being
      published.
- [ ] Third-party dependencies and external repositories are attributed.

## Reproducibility

- [x] The production notebook checksum matches the intended reference release.
- [x] Every canonical trainer starts and displays `--help` in a clean
      environment.
- [ ] Dataset identifiers and artifact hashes resolve for an authorized user.
- [x] Training, OOF, held, practice, and hidden measurements are labeled
      separately.
- [x] No unannotated population is represented as confirmed negative labels.
- [x] Known train/serve graph differences remain documented.

## Public presentation

- [x] The architecture PDF renders correctly on all 25 pages.
- [x] README links resolve from the GitHub repository root.
- [x] Historical filenames are explained rather than presented as current
      scientific claims.
- [x] No credentials, personal paths, generated caches, or private notes are
      present.
- [x] `python tools/verify_repository.py` passes.
- [x] `SHA256SUMS.csv` is regenerated after the final approved edit.

Publishing and pushing remain separate, deliberate actions after review.
