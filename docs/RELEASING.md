# Releasing on GitHub and Zenodo

Zenodo archives every GitHub release of the repository and gives it a DOI.
The metadata of the archive are in `.zenodo.json` (title, authors, license,
keywords, description); `CITATION.cff` gives the citation shown by GitHub.

## First release

1. Create the repository on GitHub (public) and push the code:
   ```bash
   git init && git add -A && git commit -m "Release 1.0.0"
   git branch -M main
   git remote add origin git@github.com:<user>/mapf-dp-local-search.git
   git push -u origin main
   ```
   The workflow in `.github/workflows/tests.yml` builds the code and runs the
   tests on every push.
2. On https://zenodo.org, log in with GitHub, open *GitHub* in the account
   menu, and switch on the repository.
3. On GitHub, create a release: *Releases* -> *Draft a new release*, tag
   `v1.0.0`, title `1.0.0`, and the notes of `CHANGELOG.md`. Publishing it
   makes Zenodo archive the repository and assign a DOI (a few minutes).
4. Copy the DOI of the record from Zenodo, and put it
   - in the badge at the top of `README.md` (replace `XXXXXXX`);
   - in `CITATION.cff`, as
     ```yaml
     identifiers:
       - type: doi
         value: 10.5281/zenodo.XXXXXXX
     ```
   - in the paper (Data and code availability), next to the URL of the repository.

   Commit these changes; they will be in the next release.

## Animations on GitHub Pages

The GIF in the README is shown by GitHub directly. The interactive animations
(`docs/animations/*.html`) are published by GitHub Pages: in the repository,
*Settings* -> *Pages* -> *Build and deployment*: source *Deploy from a branch*,
branch `main`, folder `/docs`. After a minute the site is at
`https://<user>.github.io/mapf-dp-local-search/` (`docs/index.html`), and every
page of `docs/animations/` at `.../animations/<name>.html`. To add an
animation, generate it with `mapf_bench.visualize --out docs/animations/<name>.html`
(and `--gif` for a preview in the README), link it from `docs/index.html`, and
push. `docs/.nojekyll` makes Pages serve the files as they are.

## Later releases

Update `__version__` in `mapf_bench/__init__.py`, `version` and
`date-released` in `CITATION.cff`, `version` in `.zenodo.json`, and
`CHANGELOG.md`, then publish a new GitHub release with tag `vX.Y.Z`. Zenodo
creates a new version of the same record: each version has its own DOI, and
the *concept DOI* of the record always points to the latest one (cite the
concept DOI for the software in general, the version DOI for the exact code of
a paper).

## Before the first release

- Replace `<user>` in `CITATION.cff` (`repository-code`), and in the paper.
- Check the authors and affiliations in `.zenodo.json` and `CITATION.cff`; add
  ORCID identifiers if available (`"orcid": "0000-0000-0000-0000"` in
  `.zenodo.json`, `orcid: https://orcid.org/...` in `CITATION.cff`).
- `license` in `.zenodo.json` uses the SPDX identifier `GPL-3.0-or-later`;
  check on the record that Zenodo shows the GNU GPL v3.0 or later.
