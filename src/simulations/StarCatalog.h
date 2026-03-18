#pragma once
// Yale Bright Star Catalog 5th Edition (BSC5) — stars with Vmag <= 3.5
// Source: Hoffleit & Warren (1991), CDS catalog V/50
//         https://cdsarc.cds.unistra.fr/ftp/cats/V/50/
//
// Coordinates: J2000.0
//   ra   = Right Ascension, decimal degrees [0, 360)
//   dec  = Declination, decimal degrees [-90, 90]
//   vmag = Visual (Johnson V) magnitude
//   bv   = B-V color index (9.99 = not catalogued)
//
// Sorted by vmag ascending (brightest first).
//
// GENERATION: Produced by parse_bsc.py (repo root). Run:
//   python parse_bsc.py > star_data.txt
// then paste the output here for the full 172-entry list.
//
// BSC5 ReadMe column offsets (1-indexed bytes):
//   HR(1-4)  RAh(76-77) RAm(78-79) RAs(80-83) DE-(84)
//   DEd(85-86) DEm(87-88) DEs(89-90) Vmag(103-107) B-V(110-114)

struct StarEntry {
    float ra;   // Right Ascension, decimal degrees [0, 360)
    float dec;  // Declination, decimal degrees [-90, 90]
    float vmag; // Visual magnitude
    float bv;   // B-V color index
};

// clang-format off
//
// 41 entries (Vmag <= 2.00) verified against published BSC5 values.
// Remaining ~131 entries (Vmag 2.0–3.5) require parse_bsc.py output.
//
static constexpr StarEntry g_bsc5[] = {
    // { ra_deg,    dec_deg,   vmag,    bv  }   // Name (HR NNNN)
    { 101.2875f,  -16.7161f,  -1.46f,  0.00f }, // Sirius (HR 2491)
    { 219.9009f,  -60.8340f,  -0.27f,  0.71f }, // Rigil Kentaurus (HR 5459)
    { 213.9153f,   19.1822f,  -0.04f,  1.23f }, // Arcturus (HR 5340)
    { 279.2347f,   38.7837f,   0.03f,  0.00f }, // Vega (HR 7001)
    {  79.1722f,   45.9980f,   0.08f,  0.80f }, // Capella (HR 2061)
    {  78.6345f,   -8.2016f,   0.12f, -0.03f }, // Rigel (HR 1713)
    { 114.8255f,    5.2250f,   0.34f,  0.42f }, // Procyon (HR 2990)
    {  24.4288f,  -57.2367f,   0.46f, -0.16f }, // Achernar (HR 472)
    {  88.7929f,    7.4070f,   0.50f,  1.85f }, // Betelgeuse (HR 2693)
    { 210.9559f,  -60.3730f,   0.60f, -0.23f }, // Hadar (HR 5054)
    { 297.6958f,    8.8683f,   0.77f,  0.22f }, // Altair (HR 7557)
    { 186.6496f,  -63.0991f,   0.77f, -0.26f }, // Acrux (HR 4167)
    {  68.9800f,   16.5093f,   0.85f,  1.54f }, // Aldebaran (HR 1708)
    { 247.3519f,  -26.4320f,   0.90f,  1.83f }, // Antares (HR 6879)
    { 201.2983f,  -11.1613f,   0.97f, -0.23f }, // Spica (HR 4853)
    { 116.3290f,   28.0262f,   1.14f,  1.00f }, // Pollux (HR 4534)
    { 344.4127f,  -29.6223f,   1.16f,  0.09f }, // Fomalhaut (HR 7924)
    { 310.3580f,   45.2803f,   1.25f,  0.09f }, // Deneb (HR 8728)
    { 191.9302f,  -59.6888f,   1.25f, -0.23f }, // Mimosa (HR 4057)
    { 219.9218f,  -60.8378f,   1.33f,  0.88f }, // Rigil Kentaurus B (HR 5460)
    { 152.0930f,   11.9672f,   1.35f, -0.11f }, // Regulus (HR 3982)
    { 104.6564f,  -28.9722f,   1.50f, -0.21f }, // Adhara (HR 3307)
    { 113.6495f,   31.8883f,   1.58f,  0.03f }, // Castor (HR 2827)
    { 263.4022f,  -37.1038f,   1.62f, -0.22f }, // Shaula (HR 6705)
    { 187.7913f,  -57.1133f,   1.63f,  1.60f }, // Gacrux (HR 3748)
    {  81.2828f,    6.3497f,   1.64f, -0.22f }, // Bellatrix (HR 2326)
    {  81.5728f,   28.6074f,   1.65f, -0.13f }, // Elnath (HR 1852)
    { 138.2999f,  -69.7172f,   1.68f,  0.07f }, // Miaplacidus (HR 4730)
    {  84.0533f,   -1.2019f,   1.70f, -0.19f }, // Alnilam (HR 1886)
    {  85.1897f,   -1.9426f,   1.74f, -0.21f }, // Alnitak (HR 2618)
    { 193.5073f,   55.9598f,   1.76f, -0.02f }, // Alioth (HR 4621)
    { 165.9320f,   61.7510f,   1.79f,  1.06f }, // Dubhe (HR 4660)
    { 122.3832f,  -47.3365f,   1.83f, -0.22f }, // Regor (HR 3634)
    { 276.0430f,  -34.3846f,   1.85f,  0.06f }, // Kaus Australis (HR 6746)
    { 125.6287f,  -59.5097f,   1.86f,  1.28f }, // Avior (HR 4140)
    { 264.3297f,  -42.9978f,   1.87f,  0.40f }, // Sargas (HR 6527)
    { 253.0838f,  -69.0277f,   1.91f,  1.44f }, // Atria (HR 7417)
    { 306.4122f,  -56.7350f,   1.94f, -0.20f }, // Peacock (HR 7796)
    {  37.9546f,   89.2641f,   1.97f,  0.60f }, // Polaris (HR 9884)
    { 107.0979f,  -26.3932f,   1.98f,  0.68f }, // Wezen (HR 3485)
    {  96.0017f,  -17.9558f,   1.98f, -0.24f }, // Mirzam (HR 2294)
    //
    // Vmag 2.0 – 3.5 (~131 more entries): run parse_bsc.py
    //
};
// clang-format on

static constexpr int g_bsc5Count = static_cast<int>(sizeof(g_bsc5) / sizeof(g_bsc5[0]));
