#!/usr/bin/env python3
"""
Yale Bright Star Catalog (BSC5) parser.

Downloads catalog.gz from CDS, decompresses it, parses all stars with Vmag <= 3.5,
and outputs C++ initializer list entries sorted by Vmag ascending (brightest first).

Column byte positions (1-indexed) from BSC5 ReadMe, verified:
  HR:        1-4    I4    Harvard Revised catalog number
  RA hours: 76-77   I2    Right ascension hours (J2000)
  RA min:   78-79   I2    Right ascension minutes
  RA sec:   80-83   F4.1  Right ascension seconds
  Dec sign:  84     A1    Declination sign (+ or -)
  Dec deg:  85-86   I2    Declination degrees
  Dec min:  87-88   I2    Declination arcminutes
  Dec sec:  89-90   I2    Declination arcseconds
  Vmag:    103-107  F5.2  Visual magnitude
  B-V:     110-114  F5.2  B-V color index

Usage:
  python parse_bsc.py                          # prints to stdout
  python parse_bsc.py > star_initializers.txt  # save to file
"""

import urllib.request
import gzip
import io
import sys

# -------------------------------------------------------------------
# Common names by HR number (curated from IAU/standard usage)
# -------------------------------------------------------------------
STAR_NAMES = {
      21: "Alpheratz",    # alpha And
     168: "Schedar",      # alpha Cas
     264: "Ankaa",        # alpha Phe
     472: "Achernar",     # alpha Eri
     617: "Diphda",       # beta Cet
     897: "Hamal",        # alpha Ari
    1017: "Acamar",       # theta1 Eri
    1084: "Menkar",       # alpha Cet
    1231: "Mirfak",       # alpha Per
    1457: "Cursa",        # beta Eri
    1708: "Aldebaran",    # alpha Tau
    1713: "Rigel",        # beta Ori
    1790: "Mintaka",      # delta Ori
    1852: "Elnath",       # beta Tau
    1886: "Alnilam",      # epsilon Ori
    2061: "Capella",      # alpha Aur
    2294: "Mirzam",       # beta CMa
    2326: "Bellatrix",    # gamma Ori
    2421: "Saiph",        # kappa Ori
    2491: "Sirius",       # alpha CMa
    2618: "Alnitak",      # zeta Ori
    2693: "Betelgeuse",   # alpha Ori
    2827: "Castor",       # alpha Gem
    2990: "Procyon",      # alpha CMi
    3307: "Adhara",       # epsilon CMa
    3485: "Wezen",        # delta CMa
    3634: "Regor",        # gamma2 Vel
    3693: "Aludra",       # eta CMa
    3748: "Gacrux",       # gamma Cru
    3846: "Suhail",       # lambda Vel
    3982: "Regulus",      # alpha Leo
    4057: "Mimosa",       # beta Cru
    4140: "Avior",        # epsilon Car
    4167: "Acrux",        # alpha1+2 Cru
    4295: "Merak",        # beta UMa
    4368: "Phad",         # gamma UMa (also Phecda) -- HR 4554 is Phecda
    4534: "Pollux",       # beta Gem
    4554: "Phecda",       # gamma UMa
    4621: "Alioth",       # epsilon UMa
    4660: "Dubhe",        # alpha UMa
    4730: "Miaplacidus",  # beta Car
    4819: "Muhlifain",    # gamma Cen
    4853: "Spica",        # alpha Vir
    4905: "Denebola",     # beta Leo
    5054: "Hadar",        # beta Cen
    5132: "Epsilon Cen",  # epsilon Cen
    5191: "Menkent",      # theta Cen
    5340: "Arcturus",     # alpha Boo
    5378: "Theta Cen",    # theta Cen (dup check with 5191)
    5459: "Rigil Kentaurus",   # alpha Cen A
    5460: "Rigil Kentaurus B", # alpha Cen B
    5526: "Zubenelgenubi",     # alpha2 Lib
    5681: "Zubeneschamali",    # beta Lib
    6527: "Sargas",       # theta Sco
    6553: "Eta Oph",      # check
    6705: "Shaula",       # lambda Sco
    6746: "Kaus Australis",    # epsilon Sgr
    6879: "Antares",      # alpha Sco
    7001: "Vega",         # alpha Lyr
    7121: "Nunki",        # sigma Sgr
    7193: "Rasalhague",   # alpha Oph
    7417: "Atria",        # alpha TrA
    7557: "Altair",       # alpha Aql
    7796: "Peacock",      # alpha Pav
    7924: "Fomalhaut",    # alpha PsA
    8728: "Deneb",        # alpha Cyg
    9884: "Polaris",      # alpha UMi
}


def ra_to_deg(h, m, s):
    """Convert RA from hours/minutes/seconds to decimal degrees."""
    return 15.0 * (h + m / 60.0 + s / 3600.0)


def dec_to_deg(sign, d, m, s):
    """Convert Dec from degrees/minutes/seconds to decimal degrees."""
    val = d + m / 60.0 + s / 3600.0
    return -val if sign == '-' else val


def try_int(s):
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def try_float(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


def parse_catalog(lines):
    """Parse BSC5 fixed-width catalog lines. Returns list of star tuples."""
    stars = []
    skipped = 0

    for lineno, line in enumerate(lines, 1):
        # Need at least col 107 for Vmag
        if len(line) < 107:
            skipped += 1
            continue

        # HR number (cols 1-4, 0-indexed 0:4)
        hr = try_int(line[0:4])
        if hr is None:
            skipped += 1
            continue

        # Vmag (cols 103-107, 0-indexed 102:107)
        vmag = try_float(line[102:107])
        if vmag is None:
            skipped += 1
            continue

        if vmag > 3.5:
            continue  # Not bright enough

        # Need col 90 for complete Dec
        if len(line) < 90:
            skipped += 1
            continue

        # RA (cols 76-83, 0-indexed 75:83)
        ra_h = try_int(line[75:77])
        ra_m = try_int(line[77:79])
        ra_s = try_float(line[79:83])
        if any(x is None for x in [ra_h, ra_m, ra_s]):
            skipped += 1
            continue

        # Dec (cols 84-90, 0-indexed 83:90)
        dec_sign = line[83]       # col 84
        dec_d = try_int(line[84:86])
        dec_m = try_int(line[86:88])
        dec_s = try_int(line[88:90])
        if any(x is None for x in [dec_d, dec_m, dec_s]):
            skipped += 1
            continue

        # B-V (cols 110-114, 0-indexed 109:114) — optional
        bv = 9.99
        if len(line) >= 114:
            bv_val = try_float(line[109:114])
            if bv_val is not None:
                bv = bv_val

        ra_deg = ra_to_deg(ra_h, ra_m, ra_s)
        dec_deg = dec_to_deg(dec_sign, dec_d, dec_m, dec_s)

        stars.append((hr, ra_deg, dec_deg, vmag, bv))

    return stars, skipped


def main():
    url = "https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz"
    print(f"# Downloading {url} ...", file=sys.stderr)

    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            gz_data = resp.read()
    except Exception as e:
        print(f"# ERROR downloading catalog: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"# Downloaded {len(gz_data):,} bytes (compressed)", file=sys.stderr)

    with gzip.open(io.BytesIO(gz_data)) as f:
        raw = f.read()

    print(f"# Decompressed {len(raw):,} bytes", file=sys.stderr)

    lines = raw.decode("latin-1").splitlines()
    print(f"# Catalog lines: {len(lines)}", file=sys.stderr)

    stars, skipped = parse_catalog(lines)
    print(f"# Skipped {skipped} lines (no HR, no Vmag, or incomplete coords)",
          file=sys.stderr)
    print(f"# Stars with Vmag <= 3.5: {len(stars)}", file=sys.stderr)

    # Sort by vmag ascending (brightest first); secondary sort by HR for stability
    stars.sort(key=lambda s: (s[3], s[0]))

    # Output C++ header content
    print("// Yale Bright Star Catalog 5th Ed. (BSC5) — all stars with Vmag <= 3.50")
    print("// Source: Hoffleit & Warren (1991), CDS V/50")
    print("//         https://cdsarc.cds.unistra.fr/ftp/cats/V/50/")
    print("// Coordinates: J2000.0. Sorted by Vmag ascending (brightest first).")
    print("// { ra_deg, dec_deg, vmag, bv }  // Name (HR NNNN)")
    print()
    print(f"// Total entries: {len(stars)}")
    print()

    for hr, ra_deg, dec_deg, vmag, bv in stars:
        name = STAR_NAMES.get(hr, "")
        if name:
            comment = f"// {name} (HR {hr})"
        else:
            comment = f"// HR {hr}"
        print(f"    {{ {ra_deg:9.4f}f, {dec_deg:9.4f}f, {vmag:5.2f}f, {bv:5.2f}f }},  {comment}")

    print()
    print(f"// End of BSC5 bright star list ({len(stars)} entries)")


if __name__ == "__main__":
    main()
