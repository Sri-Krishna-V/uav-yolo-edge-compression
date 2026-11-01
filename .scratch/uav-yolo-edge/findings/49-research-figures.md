# Findings: the research basis for the paper's figures

Ticket: #49 (parent #1; related #18, #46, #47, #48).
Researched: 2026-11-01. Four parallel readers covered: (1) venue rules, (2) matplotlib, (3) how to show each kind of
result, and (4) an audit of this repo. I merged their reports and re-checked the load-bearing facts myself. Nothing under
`study/` was changed, no figure was re-rendered, and no issue was opened. Probe scripts are in the session scratchpad, not
the repo.

**Question.** What must the figures satisfy (venue rules), how do we build them reproducibly in matplotlib without a TeX
install, and how should each result be drawn? The figures are built tomorrow morning from section 9.

**Verdict.**
- No current figure can go into the paper as it is. Each one is drawn 8–13 in wide with 8.5–11.5 pt text. Scaled to a
  column, that text prints at 2–5 pt, and every venue's floor is 8 pt.
- Every PDF embeds Type 3 fonts.
- Encodings conflict between figures: hollow, colour and marker each mean different things in different figures.
- Two figures the plan implies are missing: a Result D figure and a small-object figure. The #48 intervals are also missing.
- **The fix is one style module (`studyfig.py`).** It draws every figure at its printed size: 88 mm single column,
  174 mm double. All text is at least 8 pt, in Arial, with TrueType-embedded fonts and a PDF that does not change between
  runs. It holds one global encoding: colour is the model, marker is the configuration, hollow means untrained input
  resolution only.
- **The figures are then rebuilt to the specs in section 9.** The biggest change is to Result A. A forest plot of each
  pair's AP-small difference with its bootstrap interval carries the claim, and the scatter becomes context.

Contents: 1 Sources and how far they were verified · 2 Venue rules · 3 One default for all venues · 4 Matplotlib facts ·
5 How to show each kind of result · 6 Audit of today's figures · 7 (a) The shared style spec · 8 Global encoding ·
9 (b) Spec per figure · 10 (c) Decisions for the dev · 11 Unverified items

---

## 1. Sources and how far they were verified

| Source | How it was read | Status |
|---|---|---|
| IEEE Author Center, "Create Graphics for Your Article" (pages "Resolution and Size", "File Formatting", "Making Line Graphs More Accessible") | WebFetch (a model summary of the page, not raw text) | verified, as summarised |
| IEEE Transactions template and instructions PDF, `sscs.ieee.org/images/files/Transactions-template-and-instructions-on-how-to-create-your-article-formatted_4.pdf`, sections IV.B–IV.L | downloaded and read | verified |
| `IEEEtran.cls` V1.8b (CTAN) | downloaded and read | verified |
| Springer, Applied Intelligence submission guidelines, `https://link.springer.com/journal/10489/submission-guidelines`, section "Artwork and Illustrations Guidelines" | WebFetch | verified, as summarised |
| MDPI Sensors, `https://www.mdpi.com/journal/sensors/instructions`, section "Preparing Figures, Schemes and Tables" | **read in a browser by me** (WebFetch got HTTP 403) | verified, verbatim |
| MDPI Drones, `https://www.mdpi.com/journal/drones/instructions`, section "Resolution, color and format" | read in a browser by me | verified, verbatim |
| MDPI Author Layout Style Guide, `https://www.mdpi.com/authors/layout`, the Figures section | read in a browser by me | verified, verbatim |
| `mdpi.cls` (11 Mar 2023) via the mirror `github.com/ihrke/mdpi`; Overleaf MDPI template `overleaf.com/latex/templates/mdpi-article-template/fcpwsspfzsph` | read | the class is a third-party mirror (secondary) |
| Published PDFs in `papers/` and `s10489-026-07416-6.pdf` | measured | measurements, not rules |
| matplotlib 3.11.2 installed source, `study/.venv/Lib/site-packages/matplotlib/` (below: `MPL/`), and matplotlib.org docs | read, plus probes | verified, probe-checked |
| Rougier, Droettboom & Bourne 2014, "Ten Simple Rules for Better Figures", PLOS Comp Biol, doi 10.1371/journal.pcbi.1003833 | PLOS page | verified |
| Weissgerber et al. 2015, "Beyond bar and line graphs", PLOS Biol, doi 10.1371/journal.pbio.1002128 | PLOS page | verified |
| Wong, "Points of View" columns, Nat Methods 7–8 (2010–2011) | the Wong collection PDF | verified |
| Ho et al. 2019, "Moving beyond P values: data analysis with estimation graphics" | **bioRxiv preprint** 10.1101/377978v2 only | the journal version was not read |
| Cleveland & McGill 1984, "Graphical Perception", JASA 79(387) | **not opened** (every host refused) | known only through Wong's "Design of data figures", Table 1, which is adapted from it |
| Reddi et al. 2020, "MLPerf Inference Benchmark", arXiv:1911.02549 | PDF | verified |
| Wang et al. 2024, YOLOv10, arXiv:2405.14458, Fig. 1 | PDF | verified |
| Krzywinski & Altman, "Points of Significance" (box plots, error bars) | **gated, not read** | unverified |

---

## 2. Venue rules

IEEE Access and IEEE Transactions follow one IEEE graphics policy. I found no figure rule specific to Access.

| Venue | Single column | Double / full width | Smallest font | Line weight | Vector | Raster | Colour and greyscale | Caption | Panel labels |
|---|---|---|---|---|---|---|---|---|---|
| IEEE Access and Transactions | 3.5 in / 88.9 mm / 21 pc | 7.16 in / 182 mm / 43 pc | "approximately 8- to 10-point" (template IV.I); "about 9–10 points" (File Formatting) | no minimum stated | PS, EPS, PDF; fonts embedded or outlined | colour and grey >300 dpi; line art >600 dpi | RGB or CMYK. Colour is free online; print is greyscale, or $275 per graphic for colour. Check the figure on a greyscale printout. | below the figure, "Fig. 1." | (a) (b) (c), centred below each subfigure, 8 pt Times |
| Springer Applied Intelligence | 84 mm (the rule for double-column journals). AI itself is set in one 174 mm column. | 174 mm, max height 234 mm | "about 2–3 mm (8–12 pt)" | "at least 0.1 mm (0.3 pt)" | EPS preferred; fonts embedded | 1200 dpi for line art; 600 for combination art; 300 for halftone | RGB 8 bit. Colour is free online and paid in print. The main information must survive black and white. Do not refer to colour in captions. | in the manuscript text, not in the file; bold "**Fig. 1**", no full stop | lowercase a, b, c |
| MDPI Sensors | no column rule. The text column is 138.6 mm and the template example uses 8 cm. | about 184.6 mm through `adjustwidth` | none stated for figures (tables: "no smaller than 8 pt") | none stated | the instructions name raster only; the layout guide says "any common figure formats" | "preferably no less than 600 dpi" | "RGB at 8-bit per channel", "no additional cost" for colour; no greyscale rule | bold "**Figure 1.**" below the figure; "Do not abbreviate … Figure to … Fig." | a, b, c, each described in the caption |
| MDPI Drones | same template and class | same | same | same | "in PNG, JPEG or TIFF formats", supplied "in a single zip archive" | "preferably no less than 600 dpi" | same as Sensors | same | same |

### IEEE (Access and Transactions)

- **Widths.** Author Center, "Resolution and Size":
  - "One column width: 3.5 inches, 88.9 millimeters, or 21 picas"; "Two columns width: 7.16 inches, 182 millimeters, or
    43 picas".
  - Do not size a figure narrower than one column.
  - The template, section IV.D, gives 88 mm and 181 mm and a maximum depth of 8.5 in (216 mm) including the caption.
  - `IEEEtran.cls` in journal mode: `\columnwidth` is 21 pc = 252 pt (88.6 mm) and `\textwidth` is 43 pc = 516 pt
    (181.4 mm). `\figurename` is "Fig.".
- **Resolution.** "Resolution and Size": "non-vector graphics for color and grayscale images are >300dpi"; black-and-white
  line art ">600dpi". Template IV.B: a multipart figure meets the stricter of the two.
- **Fonts.** "File Formatting":
  - Embed or outline the fonts. The recommended fonts are Helvetica, Times New Roman, Arial, Cambria and Symbol.
  - "Type should be about 9–10 points at full size."
  - Template IV.I says "approximately 8- to 10-point type".
  - Template IV.I.1: axis labels are words with the unit in parentheses, e.g. "Temperature (K)".
- **Colour.**
  - Template IV.L: colour appears on Xplore at no charge and is converted to greyscale for print.
  - "About Potential Article Processing Charges" › "Color Printing": $275 per colour graphic in print.
  - "Making Line Graphs More Accessible": use thick lines and a unique marker per series, and contrast in both colour and
    brightness. "Evaluate your graphs by checking whether they can be interpreted correctly from a greyscale printout."
    Label lines directly rather than relying on a colour key.
- **Captions.** Template IV.K: "Place figure captions below the figures"; "Do not include captions as part of the
  figures"; no borders. IV.J: write "Fig." even at the start of a sentence.
- **Not verified.** The IEEE Graphics Analyzer pages returned 404, so its checks are unknown.

### Springer, Applied Intelligence

All rules below are from `link.springer.com/journal/10489/submission-guidelines`, "Artwork and Illustrations Guidelines".

- **Widths.** "Figure Placement and Size": "84 mm (for double-column text areas), or 174 mm (for single-column text
  areas)", maximum height 234 mm.
  - Measured from the published `s10489-026-07416-6.pdf` (not a rule): the text block is 174 mm, so AI is set in a single
    column. Its Fig. 1 is 172 mm wide. Captions are 8.5 pt and the body 10 pt.
  - The `sn-jnl` template's text width with the "smallcondensed" option is **unverified**. Design figures for the 174 mm
    final size.
- **Lettering and lines.**
  - "Figure Lettering": Helvetica or Arial, "about 2–3 mm (8–12 pt)". "Do not include titles or captions within your
    illustrations."
  - "Line Art": "at least 0.1 mm (0.3 pt) wide"; "Do not use faint lines and/or lettering."
- **Formats.** "Electronic Figure Submission": "For vector graphics, the preferred format is EPS"; fonts must be embedded.
  The page does not mention PDF figures (**unverified** whether PDF is accepted, though LaTeX submissions routinely carry
  PDF).
- **Colour and accessibility.**
  - "Color Art": the print version must stay readable in black and white; "do not refer to color in the captions".
  - "Accessibility": use patterns as well as colour, and give lettering a contrast of at least 4.5:1.
- **Captions and panels.**
  - "Figure Numbering": lowercase a, b, c for parts.
  - "Figure Captions": "Identify all elements found in the figure in the figure caption".

### MDPI, Sensors and Drones (read verbatim in a browser)

- **Sensors** ("Preparing Figures, Schemes and Tables"):
  - "All figures should be of a high quality (preferably no less than 600 dpi) in PNG, JPEG or TIFF formats."
  - "Authors are encouraged to prepare figures and schemes in color (RGB at 8-bit per channel). There is no additional
    cost for publishing full-color graphics."
  - Use "the correct mathematical symbols, e.g., - instead of —".
  - "A comma should be added in numbers of five or more digits in all figures". This applies to us: "1,610 images".
  - "All figures … should have a short explanatory title and caption. Any special characters or icons in an image, e.g.,
    *, **, and #, need to have a corresponding explanation in the caption." This applies to our † and § marks.
- **Drones** has the same rules, plus figure files "must be provided during submission in a single zip archive"; "There
  should be no editable parts in the images."
- **Layout Style Guide** (`mdpi.com/authors/layout`, Figures):
  - "we recommend a minimum resolution of 600 dpi. Any common figure formats may be used"
  - "avoid missing or overlapping text. All fonts must be embedded."
  - "Add 0 before the decimal point"
  - panels "should be labeled a, b, c, d", and "Each part must be individually described in the caption"
  - "Do not abbreviate Table and Figure to Tab. or Fig."
- **Width.** `mdpi.cls` (mirror) gives a text width of 13.86 cm, and wide floats reach about 18.46 cm. Measured in
  `papers/` PDFs: figures 40–119 mm wide, with full width at 173 mm.
- **No minimum figure font or line weight** appears in any MDPI page read.

---

## 3. One default that satisfies every venue

| Item | Default | Binding rule |
|---|---|---|
| Single column | **88 mm = 3.465 in** | IEEE, 88.9 mm, and IEEE discourages narrower; it fits Springer and MDPI |
| Double column / full width | **174 mm = 6.850 in** | Springer AI's 174 mm text area; IEEE (181–182 mm) and MDPI (~184.6 mm) are wider |
| Height | at most **200 mm**, caption room included | IEEE's 216 mm including caption |
| Smallest font | **8 pt everywhere**: ticks, legend, annotations | Springer's "8–12 pt" floor and IEEE's "8- to 10-point". This overrides the 7 pt the matplotlib reader suggested. |
| Font | **Arial** (embedded TrueType) | Springer names Helvetica or Arial; IEEE lists Arial; MDPI's graphical-abstract list includes Arial |
| Line weight | axes and ticks 0.6 pt; data lines 1.25 pt; grid 0.5 pt mid-grey; nothing under 0.5 pt | Springer's 0.3 pt floor; IEEE's "thick line" advice (the pt values are my judgement) |
| Vector file | **PDF with TrueType fonts embedded** (`pdf.fonttype 42`); **EPS on request** for Springer (`ps.fonttype 42`) | IEEE accepts PDF and EPS; Springer prefers EPS |
| Raster file | **PNG at 600 dpi**, rendered at the printed size | MDPI and IEEE line art at 600; Springer combination art at 600. Springer's 1200 dpi line-art rule applies only to bitmap line art, so stay vector. |
| Transparency | **none**: use solid light tints, never `alpha` | The EPS (PostScript) backend cannot do it: `MPL/backends/backend_ps.py:443` warns "The PostScript backend does not support transparency" |
| Colour | RGB. The figure must read in greyscale; every series has a marker or linestyle as well as a colour; captions never name a colour. | Springer, which says not to refer to colour, and IEEE's greyscale printout test |
| Title and caption | none inside the figure; the caption lives in LaTeX | IEEE IV.K; Springer "Figure Lettering" |
| Panel labels | "(a)", "(b)" at each panel's top left, 8 pt bold; each panel described in the caption | Springer and MDPI. IEEE wants labels centred below subfigures, which matters only if the panels become separate `\subfloat`s. |
| Numbers | thousands comma ("1,610"), a leading 0 ("0.12"), and a real minus sign "−" (U+2212) | MDPI's figure rules |
| Interval bars | the caption says what they are (method, level, number of resamples, n) | No venue page states this; it follows Weissgerber 2015 and Rougier rule 4 |

**In LaTeX:** include each figure at its natural size (`\includegraphics{fig.pdf}` with no `width=`). Fonts then print
at exactly their designed size. In the plain `article` class (10 pt; its `\textwidth` is 345 pt = 4.77 in, which is
**from memory, unverified**), an 88 mm figure sits centred with margin to spare. A 174 mm figure is wider than 4.77 in, so
the double-column figures need the venue class or a `\makebox[\textwidth]` overhang until the venue is chosen.

---

## 4. Matplotlib facts (installed 3.11.2, `study/.venv`)

### 4.1 Exact widths
- **figsize is in inches** (docs, "Figure size in different units"). A LaTeX pt is 1/72.27 in; a PDF point is 1/72 in.
- **Probe.** `figsize=(252/72.27, 2)` gives MediaBox `[0 0 251.0585 144]`, which is 252 TeX pt. A 174 mm figure gives
  MediaBox width 493.228 (= 6.8504 × 72), checked today.
- **`bbox_inches="tight"` changes the size.** `MPL/backend_bases.py:2261-2277` re-crops to the tight box plus
  `savefig.pad_inches` (0.1 in by default, `MPL/mpl-data/matplotlibrc:744`). The same probe figure came out 259.5 ×
  152.4 instead.
- **Never use `"tight"`.** Fix the figsize and let the layout engine keep labels inside it.
- **Checking the size:** `re.search(rb"/MediaBox\s*\[[^\]]*\]", pdf_bytes)`. pypdf and PyMuPDF are not installed.

### 4.2 Font embedding
- **The default is Type 3.** `pdf.fonttype` and `ps.fonttype` default to 3 (`matplotlibrc:768-773`;
  `MPL/rcsetup.py:283` validates). `MPL/backends/backend_pdf.py:1375-1378` dispatches to `embedTTFType3` or
  `embedTTFType42`. All five current PDFs contain `/Subtype /Type3` and no `/FontFile2` (grep).
- **Type 42 embeds TrueType.** It writes a `/Type0` font with `/CIDFontType2` and `/FontFile2`, which is TrueType embedded
  (probe). Fonts are always subset, with a 6-letter prefix (`backend_pdf.py:794-815`). Docs, "Fonts in Matplotlib" ›
  "Fonts in PDF and PostScript": Type 3 has no hinting; Type 42 is TrueType.
- **Check embedding** with a regex on `/FontName /(\S+)`, `/FontFile\d?` and `/Subtype /(Type3|TrueType|Type0)`. A font
  with no `/FontFile*` is not embedded.
- **Arial may be embedded.** `C:\Windows\Fonts\arial.ttf` has OS/2 `fsType` = 8, "editable embedding" (checked today
  with fontTools). The meaning of the bit is per the OpenType spec, OS/2 table, fsType:
  `https://learn.microsoft.com/typography/opentype/spec/os2#fstype` (that page was **not fetched**; the bit meaning is
  standard). A probe PDF in Arial at fonttype 42 embedded `GCUSIN+ArialMT` with no Type 3. DejaVu Sans has `fsType` 0,
  "installable".

### 4.3 Fonts without TeX
- **usetex is not available.** It needs LaTeX, plus dvipng for the Agg backends (docs, "Text rendering with LaTeX").
- **What ships.** `MPL/mpl-data/fonts/ttf` has DejaVu, STIXGeneral (Times-like) and the cm* Computer Modern fonts. It has
  no STIX Two, Latin Modern or CMU.
- **mathtext font sets.** Docs, "Mathtext" › "Mathtext font sets": `dejavusans`, `dejavuserif`, `cm`, `stix` (blends with
  Times), `stixsans`, `custom`.
- **Choice: Arial for every figure.** Every venue accepts a sans figure font, and Springer asks for one.
  - Write "AP-small", "×", "−" and "ms" as plain Unicode text, so mathtext is never needed.
  - Do not match the body font. Matching would mean cmr10 or STIXGeneral, which are thin at 8 pt and which Springer does
    not list.
- **Fail loudly if Arial is missing.** Look it up with `font_manager.findfont("Arial", fallback_to_default=False)` and
  raise if it is not found. A silent fallback to DejaVu would change every PDF on another machine.

### 4.4 One style block
- **How style files load.** `plt.style.use()` accepts a name, a file path, a URL, a package style or a list (docs,
  "Customizing Matplotlib with style sheets and rcParams").
- **Keys a style file cannot set.** It ignores a fixed blacklist (`MPL/style/__init__.py:35-40`): `backend`,
  `interactive`, `savefig.directory`, `date.epoch` and others. `pdf.fonttype`, `savefig.*` and `axes.prop_cycle` are
  allowed.
- **`savefig(metadata=…)` is not an rcParam**, so the save function must pass it.
- **Recommendation: a dict in `studyfig.py`, applied with `mpl.rc_context()` inside each figure function.** The style
  then lives in code under test, with no `.mplstyle` file to keep in sync.

### 4.5 constrained_layout, not tight_layout
- **tight_layout.** Docs, "Tight layout guide" › "Limitations": it is "experimental" and "mildly discouraged", and
  constrained layout "should typically be used instead". It ignores `suptitle` and `fig.text`. That is why the code hand-
  tunes `rect=` (`studynms.py:311,340`; `studypairs.py:396`; `studytables.py:551`).
- **constrained layout.** Docs, "Constrained layout guide":
  - Enable it with `layout="constrained"`.
  - It handles colorbars and suptitles.
  - `fig.legend(loc="outside lower center")` reserves space for the legend (`MPL/legend.py:330-344`, the `outside`
    locations).
- **Probe (174 × 56 mm, two panels, outside legend).** The legend's box ended at y = 0.113 of the figure height and both
  axes started at y = 0.237, so nothing overlapped. The MediaBox stayed at exactly 493.23 pt.
- **Do not mix the two engines.** Calling `tight_layout` switches constrained layout off.

### 4.6 Deterministic files (probe-verified)
- **What varies in the PDF.**
  - `_create_pdf_info_dict` (`backend_pdf.py:115-190`) writes `Creator`, `Producer` and `CreationDate`, and a user
    metadata value of `None` deletes the key (line 160).
  - `SOURCE_DATE_EPOCH` is honoured (lines 146-153).
  - The trailer has no `/ID` (lines 1862-1871).
  - Subset prefixes hash a frozenset of ints, which is not randomised by `PYTHONHASHSEED`.
- **Probe.** The same figure was saved three times under `PYTHONHASHSEED=1,2,3` with `metadata={"CreationDate": None}`.
  The PDFs were byte-identical. Without that argument, two runs differed only in `/CreationDate`.
- **PNG** is already deterministic: the default `Software` key carries only the version (`backend_agg.py:520-524`).
- **SVG** needs `svg.hashsalt` (`backend_svg.py:340-352`, 496-498), but we do not emit SVG.
- **Remaining change on upgrade.** `Creator` and `Producer` carry the matplotlib version, so the PDFs change only when
  matplotlib is upgraded. Keep it pinned at 3.11.2.

### 4.7 Colour-blind-safe palettes
- **What ships in `MPL/mpl-data/stylelib/`:** `tableau-colorblind10`, `petroff6`, `petroff8` and `petroff10` (Petroff
  2021, arXiv:2107.02270, which designs for colour-vision deficiency and greyscale lightness separation),
  `seaborn-v0_8-colorblind`.
- **Wong's 8-colour palette** (Nat Methods 8:441, 2011, "Color blindness", Fig. 2; usually called Okabe–Ito):
  - black (0,0,0), orange (230,159,0), sky blue (86,180,233), bluish green (0,158,115), yellow (240,228,66), blue
    (0,114,178), vermillion (213,94,0), reddish purple (204,121,167).
  - The jfly Okabe–Ito page shows it only as an image, so the hex values are from Wong's column.
- **CIE L\* of Wong's colours** (computed today): black 0, darkgrey `#555555` 36.1, blue 46.0, vermillion 54.2, bluish
  green 57.7, reddish purple 61.0, grey `#999999` 63.2, sky blue 69.8, orange 70.6, yellow 89.1.
- **Today's palette** (`studynms.py:263`, `studytables.py:492-495`), L\*: blue `#2a78d6` 50.4, orange `#eb6834` 59.7,
  green 63.6, amber 71.9, violet 43.0, pink 65.0.
  - The two model colours are only 9 L\* apart, and green, pink and amber sit within 8 L\* of each other.
  - A deuteranopia simulation merges orange with amber and blue with violet. This is the reader's Machado-matrix
    simulation, with the matrices from memory, so **approximate**.
- **Okabe–Ito advice** (jfly page): distinguish series "BOTH in color and shape", use thicker lines, and label lines
  directly.
- **Choice:**
  - Models: Wong blue `#0072B2` (L\* 46) for YOLO11s and orange `#E69F00` (L\* 71) for YOLO26s. Their 25 L\* gap
    survives a greyscale print.
  - Every other series is told apart by marker and linestyle as well (section 8).

---

## 5. How to show each kind of result (literature)

- **The reference papers have no accuracy–latency figure.**
  - Sensors 25:4857 has architecture diagrams and detections. Its speed is a table.
  - Algorithms 19:737 reports size-stratified AP only as a table (its Table 3), and its FPS only on an RTX 4090.
  - The Applied Intelligence review asks for median, P95 and P99 latency. Its Appendix Table 8 asks for network,
    post-processing and end-to-end time "reported separately", which supports our Result B. It has no plotting convention.
  - So our accuracy–latency figure is new relative to these papers. The only conventions to borrow come from the next two
    items.
- **YOLOv10 Fig. 1** (arXiv:2405.14458): latency–AP and size–AP panels with linear x, one dashed line per model family
  through its scale points, a legend box, and the authors' family as a thick line. No Pareto polyline; the family lines
  act as the front.
- **MLPerf Inference** (arXiv:1911.02549):
  - Accuracy is a gate (99 % of FP32, §IV-B), not an axis.
  - Single-stream latency is the 90th percentile (Eq. 1–2, Table IV).
  - Fig. 8 uses a **log y** axis for 100× spreads.
  - Fig. 1 is an accuracy–GFLOPs bubble chart. Its text says the trade-off "creates a Pareto frontier", but the frontier
    is not drawn.
- **Perceptual ranking.** Cleveland & McGill 1984, via Wong, "Design of data figures", Nat Methods 7:665, Table 1. From
  best to worst: position on a common scale, position on non-aligned scales, length, angle or slope, area, volume or
  saturation, hue.
  - The same column: "it can be tough to accurately judge the differences between two curves".
  - So the pair difference that Result A claims should be drawn as a position on its own axis, not read as a gap between
    markers in a busy scatter.
- **Paired data.**
  - Weissgerber 2015, Fig. 2: show paired data as pairs, and add a panel of the differences. A bar graph "suggests that the
    groups are independent".
  - Ho et al. 2019 (preprint), Fig. 2B: the observed values next to the effect size with its bootstrapped 95 % CI, "an
    estimation plot that visualizes observed values and the effect size side-by-side". That matches #48's percentile
    bootstrap.
- **Showing distributions.** Weissgerber 2015 (Recommendations): show the data. SE "magnifies the apparent visual
  differences", so state SD or the interval.
  - No source we read argued for violins or ECDFs, so any case for them is my own reasoning.
  - Result B has about 1,610 images per cell, so a median line with an IQR band (and per-image points in a supplement) is
    defensible.
  - MLPerf's tail emphasis argues for printing p95 alongside the median, as the tables already do.
- **Small multiples.**
  - Rougier 2014 rule 7 (Fig. 6): panels with different ranges make readers "believe there are significant differences".
  - Rule 8 (Fig. 7): direct labels, no legend box.
  - Rule 4: the caption says how to read the figure.
  - Wong & Shoresh, "Data exploration", Nat Methods 9:5 (2012): "using a common scale allows the plots to be directly
    compared."
  - For Pi 5 against the phone, give the **AP-small axes the same range** and let latency differ (Pi 5 112–1,695 ms, phone
    247–763 ms, from `cells.csv`). Say in the caption that the latency axes differ.
- **Provisional data.** No source prescribes a marking. The closest precedent is MLPerf's status categories (Available,
  Preview, RDI) and its change log, carried with each result.
  - Wong, "Salience to relevance" (Nat Methods 8:889): salience should match relevance, so provisional points should not
    be the most salient thing on the page.
  - Hollow is already taken. Mark the phone with its own panel or figure, a "provisional latency" tag in the panel, a
    horizontal bar of ±5 % on its latency (the drift bound; AP is unaffected), and caption wording.
- **Colour.**
  - Wong, "Color coding", Nat Methods 7:573: 6–8 colours at most, varied in lightness and saturation as well as hue, plus
    shape beyond that.
  - "Avoiding color", 8:525: map quantities to position, not colour.
  - "Arrows", 8:701: keep arrowheads for direction (the pair arrows are a fair use).
  - "Simplify to clarify", 8:611: the fewest marks; move repeated words into a header.

---

## 6. Audit of today's figures

### 6.1 Code conventions
- **No shared style.** There is no `rcParams`, `style.use`, `pdf.fonttype` or `bbox_inches` anywhere in `scripts/`.
  `save_figure` (`studynms.py:344-349`) runs `savefig(pdf)` and `savefig(png, dpi=200)` with no metadata, so every PDF has
  Type 3 fonts and a fresh `CreationDate`.
- **Layout.** Every figure uses `tight_layout(rect=…)` with hard-coded fractions, plus a suptitle and a 1–3 line `fig.text`
  subtitle that is really caption text (`studynms.py:308-311`, `studypairs.py:389-396`, `studytables.py:544-551`).
- **Axis styling is copied three times:** `studynms._axes` (`:267-277`), `studypairs.figure` (`:371-378`) and
  `studytables.figure` (`:530-536`).

| Figure | figsize (in) | MediaBox (pt) | Ticks at 88 mm / 174 mm | Annotations at 88 / 174 |
|---|---|---|---|---|
| `result_a_pi5` (`studypairs.py:349`) | 13 × 8.2 | 936 × 590.4 | 2.7 / 5.3 pt | 7.5 pt → 2.0 / 4.0 |
| `result_a_m34` | 11 × 6 | 792 × 432 | 3.2 / 6.2 | 7.5 pt → 2.4 / 4.7 |
| `result_b_nms` (`studynms.py:294`) | 11 × 4.6 | 792 × 331.2 | 3.2 / 6.2 | legend 9 pt → 2.8 / 5.6 |
| `result_b_nms_pooled` (`studynms.py:327`) | 11 × 4.4 | 792 × 316.8 | 3.2 / 6.2 | — |
| `result_c_pi5` (`studytables.py:512`) | 8.2 × 6.4 | 590.4 × 460.8 | 4.2 / 8.3 | 8 pt → 3.4 / 6.7 |

(The 10 pt default tick labels, scaled to the default widths of section 3.)

### 6.2 Per figure
- **`result_a_pi5`**: three runtime panels of AP-small against median latency, a line through each model's baselines,
  arrows for the headline pairs, and hollow markers for untrained input resolutions.
  - In (a), the "800", "960" and "1120" labels collide with the arrows and markers.
  - In (b) and (c), the pruned pairs (0.0 % and 4.0 % apart) are near-coincident points, so their arrows vanish.
  - INT8 gains and pruning losses use the same arrow.
  - B3 INT8 at 1280 has no arrow and nothing says why (14.1 % apart).
  - The x label says "at 2000 MHz" (`studypairs.py:370`), yet nothing marks throttling. Every ONNX Runtime FP32 baseline
    in a headline pair throttled (`equal_latency.csv`, `baseline_throttled`).
  - The caption sentence about the sessions is a hard-coded literal (`studypairs.py:393`).
  - The legend is 4 × 3.
- **`result_a_m34`**: one sparse panel with no ladder of baselines and one arrow (B1 INT8 at 960 against A1 FP32 at 640,
  8.6 % apart). The drift (+4.9 %, +3.4 %) and the provisional status are nowhere in the figure. #46 (1 Nov) says it is
  redrawn when phone sessions 3 and 4 land.
- **`result_b_nms`**:
  - One linear y axis from 0 to ~22 ms crushes the 0.25-threshold curves (~1 ms) and the NMS-free head (~0.2 ms) against
    zero.
  - The title shows a run id (`studynms.py:308`, `pair['run']`).
  - `--figure-cell` defaults to the **stock** INT8 cell, `pi5-matrix-2:onnxruntime-int8` (`studynms.py:363`). INVARIANTS
    §8 makes per-channel the matrix's INT8, and that pair exists (session 7, `RESULTS.md:127`).
- **`result_b_nms_pooled`**: 26 one-to-many cells as grey lines at alpha 0.35 (EPS cannot show alpha), unshared y, and the
  legend in one panel only. It belongs in the supplement.
- **`result_c_pi5`**: the latency ratio YOLO26s/YOLO11s against input resolution, one line per runtime.
  - Hollow means "throttled" here but "untrained input resolution" in A.
  - Stock and per-channel INT8 are the same pink, and the stock line is hidden behind the other.
  - Green, pink and amber are close in greyscale.
  - The legend carries data values, and the label "equal latency" misuses a glossary term.
  - The legend numbers do agree with `RESULTS.md:195-196,222-224`.

### 6.3 Inconsistencies between figures
- **Colour.** Colour is the model in A, the head in B (`studynms.py:298`) and the runtime in C.
- **Marker.** Marker is the configuration in A, the threshold in B and the runtime in C.
- **Line style.** Dashed is the 0.25 threshold in B but the stock recipe in C.
- **Hollow** is untrained input resolution in A but throttled in C.
- **Grids** differ: y only in B, both axes in A and C.
- **`RESULTS.md` is stale against `cells.csv`.**
  - It says the phone has 2 cells and no figure (`RESULTS.md:14-15,104,387,420`) and has a "2.09×" stock phone speed-up
    (`:293`).
  - `cells.csv` now has 88 rows, 14 of them the phone's.
  - `result_d_int8.csv` gives the phone a per-channel speed-up of 2.01× (A1) and 1.93× (A2), and 2.22× stock.
  - Captions must be regenerated from the tables, not copied from `RESULTS.md`.

### 6.4 The figure plan on the issues
- #18 section 5 lists three outputs: AP-small against latency, "one figure for each platform (the main figure)"; a table of
  pairs; NMS time against objects per image on the Pi 5.
- #46 adds: panels per runtime, both models, marker = configuration, a line through the baselines, a connector per pair,
  and the phone "if it has three or more cells".
- #47 adds the Result C ratio figure. #49 asks for Results A–D "with the tables and figures", without naming figures.
- No Result D or small-object figure is planned. The #48 intervals are **not computed yet**:
  - `equal_latency.csv` and `small_object.csv` have no filled CI columns.
  - `48-bootstrap/` holds only probes.
  - The production script is "still to write" (#48, 1 Nov).

---

## 7. (a) The shared style spec: `study/scripts/studyfig.py`

One module, imported by `studypairs.py`, `studynms.py` and `studytables.py`. It replaces `studynms.save_figure`, the
palette constants and the three `_axes` copies. A sketch, not yet written to the repo:

```python
"""The paper's figure style: widths, fonts, the global encoding, and a deterministic save."""
from contextlib import contextmanager
from pathlib import Path

MM = 1 / 25.4
SINGLE = 88 * MM          # 3.465 in: IEEE's column, inside Springer's and MDPI's
DOUBLE = 174 * MM         # 6.850 in: Springer AI's text width, inside IEEE's and MDPI's
MAX_HEIGHT = 200 * MM
PNG_DPI = 600

FONT = "Arial"            # fsType 8, editable embedding; Springer and IEEE both name it
SIZE = 8                  # the floor for every piece of text (Springer 8-12 pt; IEEE 8-10 pt)

# The global encoding (section 8). Colour is the model; marker is the configuration.
MODEL_COLOR = {"YOLO11s": "#0072B2", "YOLO26s": "#E69F00"}            # Wong blue / orange, L* 46 / 71
CONFIG_MARKER = {"baseline": "o", "quantized": "s", "pruned": "^", "distilled": "D", "combined": "P"}
RUNTIME_STYLE = {                                                        # Result C and D only, where model is not a series
    ("onnxruntime", "FP32"): dict(color="#000000", ls="-"),
    ("openvino", "FP32"):    dict(color="#009E73", ls="--"),
    ("ncnn", "FP16"):        dict(color="#CC79A7", ls="-."),
    ("onnxruntime", "INT8"): dict(color="#D55E00", ls=":"),
}
HEAD_STYLE = {"one-to-many": dict(color="#000000", marker="o"), "NMS-free": dict(color="#CC79A7", marker="s")}
THRESHOLD_LS = {0.001: "-", 0.25: "--"}   # evaluation / deployment threshold
INK, MUTED, RULE, TINT = "#222222", "#555555", "#BBBBBB", "#E8E8E8"   # TINT replaces every alpha (EPS has none)

RC = {
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": SIZE,
    "axes.titlesize": SIZE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE, "ytick.labelsize": SIZE,
    "legend.fontsize": SIZE, "legend.frameon": False, "legend.handlelength": 2.2,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.5, "grid.color": RULE, "axes.axisbelow": True,
    "lines.linewidth": 1.25, "lines.markersize": 4.5, "lines.markeredgewidth": 0.75,
    "axes.unicode_minus": True, "axes.formatter.use_mathtext": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "standard",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

@contextmanager
def style():
    import matplotlib as mpl
    from matplotlib import font_manager
    font_manager.findfont(FONT, fallback_to_default=False)   # raises: no silent DejaVu fallback
    with mpl.rc_context(RC):
        yield

def figure(width, height, **kw):
    """A figure at its printed size; constrained layout; never resized by savefig."""
    import matplotlib.pyplot as plt
    assert width in (SINGLE, DOUBLE) and height <= MAX_HEIGHT
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)

def panel_label(ax, letter):
    ax.text(0, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")

def save(fig, stem: Path, eps=False):
    """PDF (TrueType fonts, no CreationDate, so re-runs are byte-identical) and a 600 dpi PNG at the printed size."""
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    if eps:
        fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)
```

The tests for it in `scripts/tests/test_studyfig.py`, each a fast check on a tiny figure:
- the PDF's MediaBox width is `SINGLE × 72` or `DOUBLE × 72` to within 0.01 pt;
- `/Type3` is absent and `/FontFile2` is present;
- two saves are byte-identical;
- every `Text` artist's size is at least 8. Walk `fig.findobj(matplotlib.text.Text)`, skipping empty strings.

The Arial lookup raises if the font is missing.

**Rules that go with the module.**
- No `suptitle` and no `fig.text` captions: the caption is LaTeX.
- No `alpha`. Use `TINT` or a lighter solid colour.
- Axis labels are words with the unit in parentheses: "Median latency (ms)".
- Input resolutions print as "640", "1,120" and "1,280" (MDPI's comma).
- Every glossary term is spelled the glossary's way: "NMS-free head", "one-to-many head", "AP-small", never "e2e".
- Captions are generated next to each figure as `<stem>.caption.tex` from the same tables, so numbers never drift from
  `RESULTS.md`. This is optional but cheap.

---

## 8. Global encoding (the same in every figure)

| Channel | Means | Only exception |
|---|---|---|
| Colour | the **model**: YOLO11s blue `#0072B2`, YOLO26s orange `#E69F00` | Result C (a ratio between models) and the runtime context, where colour means runtime from `RUNTIME_STYLE`; B, where colour means head. Each such figure states it in its caption, and none of them shows model colour. |
| Marker | the **configuration**: baseline ○, quantized □, pruned △, distilled ◇, combined ✚ | B uses the head's marker |
| Hollow marker | **an input resolution the model was not trained at** (`untrained_imgsz = 1`), and nothing else | — |
| Dagger † beside a point | **the cell throttled** (`throttled = "throttled"`) | — |
| Section sign § | the pair spans session 6 (`spans_session_6 = 1`) | — |
| Linestyle | the runtime (C and D context), or the threshold (B) | — |
| Light grey tint panel and "provisional latency" tag | **phone** latency (drift checks failed, #34) | — |
| Horizontal bar ±5 % on a phone point | the drift bound on its latency | — |

In each case colour is backed by marker or linestyle, so every figure survives a greyscale print (IEEE's test and
Springer's rule).

---

## 9. (b) Spec per figure

All columns below are in `study/results/cells.csv` unless a table is named. The row filter is `head == "one-to-many"`
unless stated. Where a stock and a per-channel INT8 cell exist for the same condition, the figure shows **per-channel**
(INVARIANTS §8); stock appears only where the figure is about the recipe.

### F1. Result A, the main figure: is a compression budget better spent on input resolution?
- **Width:** double, 174 × 75 mm, two panels side by side. This replaces `result_a_pi5`.
- **(a) Context scatter, ONNX Runtime on the Pi 5**, which holds all the INT8 pairs.
  - x: `latency_ms`, **log scale**, ticks at 200, 300, 500, 700, 1,000 and 1,500. On a log axis the pair rule's ±10 %
    window has the same width everywhere. This choice is my own reasoning; MLPerf Fig. 8 is the log precedent.
  - y: `ap_small`, with limits shared by F2 and the supplement panels.
  - One line per model through its baselines (640–1,280), coloured by model. Each baseline carries its input resolution
    as an 8 pt label, placed by a fixed offset rule and checked for overlap.
  - Compressed cells use the configuration marker. A thin grey connector (no arrowhead; Wong, "Arrows") joins each
    headline pair.
  - Hollow marks an untrained input resolution, and † marks a throttled cell.
- **(b) Forest plot of the claim**, which carries the paper's message.
  - One row per **headline pair** (`tables/equal_latency.csv`, `headline == 1`; 11 on the Pi 5 across all three
    runtimes, plus 1 on the phone, which is drawn in F2, not here).
  - x: `ap_small_diff` (compressed minus baseline), with a vertical rule at 0. A dot sits on the point estimate, with a
    horizontal bar for the **#48 percentile bootstrap 95 % interval**.
  - Each row is labelled "YOLO11s · INT8 at 800 vs FP32 at 640 · ONNX Runtime", with `apart_pct` printed at the right ("6.8
    % apart"). The bootstrap covers AP only, not the latency gap.
  - Rows are grouped by runtime, then model. The dot takes the model's colour and the compressed configuration's marker.
- **Data:** `platform`, `runtime`, `model`, `configuration`, `imgsz`, `untrained_imgsz`, `precision`, `int8_recipe`,
  `latency_ms`, `ap_small`, `throttled`. Pairs come from `equal_latency.csv` (`compressed_cell`, `baseline_cell`,
  `ap_small_diff`, `apart_pct`, `headline`, `spans_session_6`), and CIs from #48's output, which has no columns yet.
- **Changes from today:**
  - 13 × 8.2 in becomes 174 × 75 mm.
  - The OpenVINO and NCNN scatter panels move to S1. Their pairs stay in (b).
  - Arrows become connectors.
  - Titles and the subtitle move to the caption.
  - Throttling is now marked.
  - The "at 2000 MHz" x label becomes "Median latency (ms)", and the clock lock goes in the caption.
- **If #48 is not done by morning**, (b) shows the point estimates without bars. The caption then states #9's three-seed
  spread in words, as #48's third acceptance box allows. Do not draw seed spread as if it were a bootstrap interval.

### F2. Result A on the phone (provisional)
- **Width:** single, 88 × 70 mm. This replaces `result_a_m34`.
- **Encoding.** As F1(a), on `platform == "m34"`, with the same y limits as F1.
  - The panel has a grey `TINT` background and a "Latency provisional (drift +3.4 %, +4.9 %)" tag.
  - Every point has a ±5 % horizontal drift bar.
  - Its headline pair or pairs (today one: B1 INT8 at 960 against A1 FP32 at 640, +0.0355 AP-small, 8.6 % apart) are drawn
    as forest rows beneath, in the same style as F1(b).
  - A pair whose `apart_pct + 5` exceeds 10 is labelled "not robust to drift". This is a proposal for the text, not a
    rule in INVARIANTS.
- **Data:** as F1, plus `drift` and `note`.
- **Depends on:** phone sessions 3 and 4 (FP32 at 800, INT8 at 1,120). The figure must be built from `cells.csv` at
  build time, never from hard-coded lists.
- **Placement:** decision 10.2.

### F3. Result B: what NMS costs, and how it grows with objects per image
- **Width:** double, 174 × 65 mm, two panels. This replaces `result_b_nms`.
- **Encoding.**
  - x: objects per image (`x_median` of each bin). y: post-processing time (ms), **log scale** from 0.1 to 30, because the
    two heads differ by about 100×.
  - Colour and marker by head: one-to-many black ○, NMS-free reddish purple □. Linestyle by threshold: solid for the
    0.001 evaluation threshold, dashed for the 0.25 deployment threshold. Each threshold is also named in a direct label
    at the line's end.
  - The median is the line, and the IQR (`q1`–`q3`) is a solid `TINT` band. No alpha.
  - (a) is the evaluation threshold and (b) the deployment threshold, or both thresholds on one log panel if the labels
    fit. Panel titles carry the threshold, as the glossary requires.
- **Cell:** decision 10.4. The recommendation is the per-channel INT8 pair (session 7), since INVARIANTS §8 makes it the
  matrix's INT8, with YOLO26s's two heads. Change the `--figure-cell` default at `studynms.py:363`.
- **Data:** `tables/result_b_bins.csv` (`session`, `run`, `variant`, `head`, `threshold`, `x_median`, `median`, `q1`,
  `q3`, `n`). Its source is `bench/pi5/**/timings.csv` (`t_post_ms`, `t_post_deploy_ms`, `n_cand`).
- **Changes from today:** a log y axis; the run id becomes "YOLO26s A2, ONNX Runtime INT8 (per-channel)" in the caption;
  shared y; no title block.
- **Optional (c):** each head's post-processing share of latency (`tables/result_b_heads.csv`, `*_post_share_pct`) as two
  dots per cell. It ties in with the Applied Intelligence review's call to report post-processing separately.

### S2. Result B pooled (supplement)
- **Width:** single, 88 × 65 mm.
- **Change from `result_b_nms_pooled`:** the per-cell lines become thin solid `#CCCCCC` instead of alpha 0.35 grey, with
  the pooled median in black. The y axis is shared with F3.

### F4. Result C: does YOLO26s's latency gap grow with input resolution?
- **Width:** single, 88 × 70 mm. This replaces `result_c_pi5`.
- **Encoding.**
  - x: `imgsz` (640, 800, 960, 1,120, 1,280). y: `latency_ratio` (YOLO26s / YOLO11s), with a rule at 1.0.
  - One line per runtime and precision, styled by `RUNTIME_STYLE` (colour plus linestyle). Markers are the configuration's
    (○ baseline, □ quantized), so the meaning matches F1.
  - **Direct end-of-line labels** with the value at 1,280 ("NCNN FP16 +9.0 %") replace the legend (Rougier rule 8).
  - A pair that is not clean (`clean == 0`, an image below 2000 MHz) gets † on the point. It is no longer drawn hollow.
  - Per-channel INT8 is the INT8 line. Stock INT8 moves to the supplement, or appears as a thin grey dotted line labelled
    "INT8, stock recipe".
- **Data:** `tables/result_c.csv` with `platform == "pi5"`: `runtime`, `precision`, `int8_recipe`, `configuration`,
  `imgsz`, `latency_ratio`, `latency_pct`, `clean`, `spans_session_6`, `headline`.
- **Changes from today:** hollow becomes †; the legend becomes labels; the y label becomes "Latency ratio, YOLO26s /
  YOLO11s"; "equal latency" becomes "ratio = 1".
- **Phone:** 6 phone pairs exist, all "not clean". Leave them out of F4 and keep them in the table.

### F5. Result D (new): what does INT8 buy, and what does it cost?
- **Width:** double, 174 × 70 mm, three aligned panels sharing one set of row labels. A Cleveland dot plot (Wong, "Design
  of data figures": position on a common scale).
- **Rows:** checkpoint × input resolution on the Pi 5, grouped by model: A1 640, B1 800, B1 960, B3 1,120, B3 1,280; A2
  640, B2 800, B2 960, B4 1,120. A phone group (A1 640, A2 640) follows, with the provisional tint.
- **(a) speed-up (×)**: `per_channel_speedup` as a filled □, and `stock_speedup` as a small ×, since hollow is reserved.
- **(b) AP change (%)**: `per_channel_ap_diff_pct` and `stock_ap_diff_pct`.
- **(c) AP-small change (%)**: `per_channel_ap_small_diff_pct` and `stock_ap_small_diff_pct`, with a rule at 0.
- **Colour** is the model.
- **Rows marked §** (`*_spans_session_6 = 1`: B3 and B4 at 1,120) keep the § and a caption note, or are dropped (decision
  10.6).
- **The message it makes visible:** on the Pi 5 the speed-up is nearly flat (1.38–1.59×). The AP-small cost varies from
  −2.5 % to −13.6 % per-channel, and stock costs YOLO26s far more (−14.3 % to −29.9 %).
- **Data:** `tables/result_d_int8.csv`: `platform`, `model`, `checkpoint`, `imgsz`, `untrained_imgsz`, `per_channel_*`,
  `stock_*`, `*_spans_session_6`.
- **Data issue to settle:** A2 at 640 has **two** Pi 5 rows. Pick one by a rule, or show both, before drawing.
- **Runtimes without INT8:** NCNN and OpenVINO have no INT8 cell on the Pi 5, so they stay in the table
  (`result_d_runtimes`) and are not drawn here.

### F6. Small-object section (new): does compression cost AP-small disproportionately?
- **Width:** single, 88 × 80 mm.
- **Encoding.**
  - x: aggregate AP drop (%) (`ap_drop_pct`). y: AP-small drop (%) (`ap_small_drop_pct`).
  - Reference lines y = x, y = 1.3x and y = 2x (#9's bands), each labelled directly.
  - Colour is the model, marker the compression (configuration).
  - The hypothesis is the distance above the diagonal. That puts it as position on a common scale, the top of
    Cleveland & McGill's ranking.
  - When #48 lands, add the bootstrap intervals as crosses (`ci_low`, `ci_high`). The relative-drop intervals need the same
    resamples as the differences, so check #48's output carries them.
- **Data:** `tables/small_object.csv` (13 rows): `platform`, `model`, `compression`, `runtime`, `int8_recipe`, `headline`,
  `ap_drop_pct`, `ap_small_drop_pct`, `ap_medium_drop_pct`, `ap_large_drop_pct`, `small_over_all`, `band`, `ci_low`,
  `ci_high`.
- **Alternative:** a profile plot, one line per configuration across small, medium and large. That makes it a single
  column with three points per line. The scatter is preferred because it tests the hypothesis directly.

### S1. Result A context on OpenVINO and NCNN (supplement)
- **Width:** double, 174 × 65 mm.
- **Encoding:** F1(a)'s encoding for the other two Pi 5 runtimes. The pruned pairs (0–4 % apart) are near-coincident
  points here, which is why they leave the main figure. Their differences are in F1(b).

---

## 10. (c) Decisions for the dev

1. **Venue.**
   - Nothing above needs it. The section 3 default (88 and 174 mm, 8 pt Arial, PDF with TrueType fonts plus a 600 dpi PNG,
     no alpha, greyscale-safe) satisfies IEEE, Springer AI, MDPI Sensors and MDPI Drones.
   - The venue then changes only the caption style ("Fig." against "Figure"), panel-label placement for IEEE subfloats, and
     whether EPS (Springer) or a PNG zip (MDPI) is uploaded.
   - **Recommendation:** build to the default now and choose the venue after the draft.
2. **Phone in the main figure or a supplement.**
   - Today the phone has one pair and no ladder of baselines, and its latency is provisional.
   - **Recommendation:** F2 is a separate single-column figure. It goes in the main text **only if** sessions 3 and 4 land
     and give at least two headline pairs that survive the ±5 % drift bound; otherwise it goes in the supplement.
   - The phone never shares F1's axes, which keeps the Pi 5's clean claim free of the provisional data.
3. **How to show intervals.**
   - **Recommendation:** a separate forest panel, F1(b), carries the AP-small difference and its 95 % percentile bootstrap
     interval. Do **not** put vertical error bars on the F1(a) points: the interval belongs to the difference, not to
     either cell.
   - The caption states "percentile bootstrap over the 1,610 test-dev images, *R* resamples, paired by image".
   - Fallback without #48: point estimates only, and the seed spread in words.
   - Choose R = 200 (about 20 min for 8 pairs) or 1,000 (about 1.5 h). The #48 estimate was for 8 pairs; there are now 12.
4. **Which INT8 cell Result B shows.** Per-channel (session 7) or stock (session 2, today's default). The Result B text in
   `RESULTS.md` quotes the stock cell, so it changes with this choice.
5. **The Result D stock speed-up on the phone** (2.22×) is a ratio across software versions. #47's comment (1 Nov) flags it
   as "needs a decision": plot it, or plot per-channel only.
6. **The § rows** (B3 and B4 at 1,120, which span session 6, where latencies may be 3–5 % high). Mark them in F1(b) and F5,
   or drop them.
7. ~~**Duplicate A2 at 640 in `result_d_int8.csv`.**~~ Not a duplicate: the two rows are the one-to-many head and the
   NMS-free head (`head` column). The figure filters `head == "one-to-many"`. (Corrected after the build, 1 Nov.)
8. **Log x on F1(a).** It makes the pair window a constant width. Linear is easier for some readers. **Recommendation:**
   log.
9. **Font.** Arial (recommended; it suits Springer and IEEE) or a body-matched serif (STIXGeneral for a Times body, cmr10
   for Computer Modern), which Springer does not list.

---

## 11. Unverified items

- `sn-jnl` with "smallcondensed": its text width.
- IEEE Graphics Analyzer: its checks (404).
- Whether Springer AI accepts PDF figures; only EPS is named.
- MDPI: any minimum font or line weight. None is stated; for tables only, "no smaller than 8 pt".
- `mdpi.cls` geometry: read from a third-party mirror.
- The plain `article` 10 pt `\textwidth` of 345 pt is from memory. The paper's real `\columnwidth` was not read, since no
  `.tex` is in the repo.
- Cleveland & McGill 1984: not opened. Its ranking is cited through Wong (2010), Table 1.
- Krzywinski & Altman's columns: not read.
- Ho et al. 2019: preprint only.
- The colour-blind simulation of today's palette: matrices from memory.
- The OpenType fsType bit meaning: standard, but the spec page was not fetched for this note.
- The Ultralytics YOLO26 comparison images: not read.
