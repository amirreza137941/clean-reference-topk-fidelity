# Anonymous artifact policy

This repository snapshot is prepared for double-blind review.

## Keep during review

- Author metadata remains `Anonymous Authors`.
- The personal upstream repository remains private.
- Reviewers receive a frozen, anonymous mirror of one exact commit.
- Source, documentation, commit messages, paths, result metadata, and archives
  are checked for identifying strings before the mirror is created.
- The anonymous mirror is not modified during review unless the venue's policy
  explicitly authorizes a revision.

## Do not add during review

- names, initials, affiliations, ORCIDs, personal email addresses, or personal
  GitHub profile links;
- acknowledgements, grant identifiers, lab server names, or private paths;
- links to public preprints or repositories that reveal authorship;
- badges whose target URL contains the personal repository owner;
- public issues, forks, stars, or social posts connecting an identity to the
  anonymous artifact.

## After review

Complete and rename `CITATION.cff.template`, update the security contact and
repository URLs, add the accepted-paper reference, and make a tagged release.
Archive the release in a DOI-granting repository rather than relying only on a
mutable branch.
