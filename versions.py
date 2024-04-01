from filenames import PathLike, getFileNames

#'SHA <-> version-number' relation
# (note that duel entry for 1.2.5/1.4.2 is due to update glitches)
supportedVersions = {
    "1446ABC": "1.2.5",
    "9BEADBE": "1.3.0",
    "2FDC90": "1.4.1",
    "928D2AE": "1.2.5",
    "3D3CFF5": "1.?.?",
    "A0F6980": "1.?.?",
    "340b03f": "1.4.2",
    "82755AE": "1.4.2",
    "B218FBD": "1.5.1",
    "E52AC4D": "1.5.2",
    "1323D38": "1.5.3",
    "73A3A4D0": "1.6.0",
    "6171C497": "1.6.2",
    "A98A2E52": "1.7.0",
    "A4FAAEBA": "1.7.1",
    "E7C7CD94": "1.8.0",
    "412432D3": "1.9.0",
    "9F438E3C": "1.9.1",
    "97A11B27": "1.10.0",
    "2992193B": "1.11.0",
    "2569BD17": "1.11.1",
    "81C1B398": "1.11.2",
    "93CE0B95": "1.12.0",
    "FE6412C3": "1.13.0",
}

# number assigned to each release, and compatibility number (each release
#    gets a new number, and shares a compatibility number with all releases
#    that can be concatenated in single output files.
#
ordinalVersionNumber = {
    "1446ABC": (0, 0),
    "9BEADBE": (1, 0),
    "2FDC90": (2, 1),
    "928D2AE": (0, 0),
    "340b03f": (3, 1),
    "82755AE": (3, 1),
    "B218FBD": (4, 2),
    "E52AC4D": (5, 2),
    "1323D38": (6, 2),
    "73A3A4D0": (7, 2),
    "6171C497": (8, 2),
    "A98A2E52": (9, 2),
    "A4FAAEBA": (10, 2),
    "E7C7CD94": (11, 2),
    "412432D3": (12, 2),
    "9F438E3C": (13, 2),
    "97A11B27": (14, 2),
    "2992193B": (15, 2),
    "2569BD17": (16, 2),
    "81C1B398": (17, 2),
    "93CE0B95": (18, 2),
    "FE6412C3": (19, 3),
}
#
# The default version number is used by spectral and location parsing routines
# as default compatibility number if none is given
#
defaultVersion = 0
defaultIIRWeightType = 0


def getVersions(path: PathLike):
    """
    This function retrieves sha from sys filenames; if no sha is present
    within the first 20 lines, it is assumed the previous found sha is
    active. The output is a version list; each entry in the version list
    is a dict that contains all file prefixes (0009 etc.) that can be
    processed in the same way (this may go across firmware versions).
    """

    # Get sys files
    path, fileNames = getFileNames(path, "SYS", "system")

    def latestVersion():
        ordinal = -1
        for key in ordinalVersionNumber:
            if ordinalVersionNumber[key][1] > ordinal:
                latVer = key
        return latVer

    # end def
    if len(fileNames) == 0:
        sha = latestVersion()
        IIRWeightType = defaultIIRWeightType
        return [
            {
                "sha": [sha],
                "version": [supportedVersions[sha]],
                "ordinal": [ordinalVersionNumber[sha]],
                "number": ordinalVersionNumber[sha][1],
                "IIRWeightType": IIRWeightType,
                "fileNumbers": [],
            }
        ]

    first = True
    version = []
    # Loop over all the _SYS files
    for index, filename in enumerate(fileNames):
        foundSha = False
        foundIIRWeightType = False
        IIRWeightType = defaultIIRWeightType
        # Check if there is a sha in first 80 lines
        with open(path / filename) as infile:
            jline = 0
            for line in infile:
                if "SHA" in line:
                    sha = line.split(":")
                    sha = sha[-1].strip()
                    foundSha = True
                elif "iir weight type" in line:
                    ___, IIRWeightType = line.split(":")
                    IIRWeightType = int(IIRWeightType.strip())
                    foundIIRWeightType = True
                jline += 1
                if (foundSha and foundIIRWeightType) or jline > 80:
                    break
        # If we found a SHA, check if it is valid
        if foundSha:
            # Is it a valid sha?
            if sha not in ordinalVersionNumber:
                # If not - parse using the latest version
                sha = latestVersion()

        # Valid sha, so what to do?
        if foundSha and first:
            # this the first file, and we found a sha
            version.append(
                {
                    "sha": [sha],
                    "version": [supportedVersions[sha]],
                    "ordinal": [ordinalVersionNumber[sha]],
                    "number": ordinalVersionNumber[sha][1],
                    "IIRWeightType": IIRWeightType,
                    "fileNumbers": [],
                }
            )
            first = False
        elif not foundSha and first:
            # this is the first file, but no sha - we will try to continue
            # under the assumption that the version corresponds to the
            # latest version - may lead to problems in older version
            print("WARNING: Cannot determine version number")
            sha = latestVersion()
            version.append(
                {
                    "sha": [sha],
                    "version": [supportedVersions[sha]],
                    "ordinal": [ordinalVersionNumber[sha]],
                    "number": ordinalVersionNumber[sha][1],
                    "IIRWeightType": IIRWeightType,
                    "fileNumbers": [],
                }
            )
            first = False
        elif foundSha and not first:
            # We found a new sha, check if it is the same as previous found sha
            # and if so just continue
            if not (
                sha in version[-1]["sha"]
                and version[-1]["IIRWeightType"] == IIRWeightType
            ):
                # if not, check if this version is compatible with the previous
                # found version
                if (
                    ordinalVersionNumber[sha][1] == version[-1]["number"]
                    and version[-1]["IIRWeightType"] == IIRWeightType
                ):
                    # If so, append the sha/version
                    version[-1]["sha"].append(sha)
                    version[-1]["ordinal"].append(ordinalVersionNumber[sha])
                    version[-1]["version"].append(supportedVersions[sha])
                else:
                    # Not Compatible, we add a new version to the version list
                    # that has to be processed seperately
                    version.append(
                        {
                            "sha": [sha],
                            "version": [supportedVersions[sha]],
                            "ordinal": [ordinalVersionNumber[sha]],
                            "number": ordinalVersionNumber[sha][1],
                            "IIRWeightType": IIRWeightType,
                            "fileNumbers": [],
                        }
                    )
        entry, ___ = filename.split("_")
        # Add file log identifier (e.g. 0009_????.csv, with entry = '0009')
        version[-1]["fileNumbers"].append(entry)
    # end for filenames
    return version
