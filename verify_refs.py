"""
Cross-check every DOI-bearing entry in references.bib against Crossref metadata.
Reports title / first-author / year / volume / issue / pages mismatches.
"""
import re, json, time, urllib.request, urllib.error, sys, os

BIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuscript", "references.bib")
MAILTO = "rachirobe@gmail.com"

def parse_bib(path):
    text = open(path, encoding="utf-8").read()
    entries = []
    # split on @type{key,
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}\s*(?:,|\Z)', body, re.S):
            fields[fm.group(1).lower()] = re.sub(r'\s+', ' ', fm.group(2)).strip()
        entries.append((key, etype, fields))
    return entries

def clean(s):
    if not s: return ""
    s = re.sub(r'[{}]', '', s)
    s = s.replace('\\&', '&').replace('--', '-')
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def first_author_family(bib_author):
    # bibtex: "Family, Given and Family2, Given2" OR "Given Family and ..."
    first = bib_author.split(' and ')[0].strip()
    first = re.sub(r'[{}\\"]', '', first)
    if ',' in first:
        return first.split(',')[0].strip().lower()
    return first.split()[-1].strip().lower()

def crossref(doi):
    url = f"https://api.crossref.org/works/{doi}?mailto={MAILTO}"
    req = urllib.request.Request(url, headers={
        "User-Agent": f"RefVerifier/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]

def cr_pages(msg):
    return msg.get("page", "")

def main():
    entries = parse_bib(BIB)
    doi_entries = [(k, t, f) for k, t, f in entries if f.get("doi")]
    print(f"Parsed {len(entries)} entries, {len(doi_entries)} with DOIs.\n")
    print("="*72)
    problems = []
    for key, etype, f in doi_entries:
        doi = f["doi"]
        try:
            m = crossref(doi)
        except urllib.error.HTTPError as e:
            print(f"[{key}] DOI {doi} -> HTTP {e.code} (DOI may not resolve!)")
            problems.append((key, f"DOI does not resolve (HTTP {e.code})"))
            time.sleep(1); continue
        except Exception as e:
            print(f"[{key}] DOI {doi} -> ERROR {e}")
            time.sleep(1); continue

        issues = []
        # title
        cr_title = clean(" ".join(m.get("title", [""])))
        bib_title = clean(f.get("title", ""))
        if bib_title and cr_title and bib_title not in cr_title and cr_title not in bib_title:
            # token overlap check
            bt, ct = set(bib_title.split()), set(cr_title.split())
            overlap = len(bt & ct) / max(len(bt), 1)
            if overlap < 0.7:
                issues.append(f"TITLE  bib='{f.get('title','')}'  crossref='{' '.join(m.get('title',['']))}'")
        # first author
        cr_auth = m.get("author", [])
        if cr_auth:
            cr_fam = clean(cr_auth[0].get("family", ""))
            bib_fam = clean(first_author_family(f.get("author", "")))
            if bib_fam and cr_fam and bib_fam not in cr_fam and cr_fam not in bib_fam:
                issues.append(f"AUTHOR1 bib='{bib_fam}' crossref='{cr_fam}'")
            # author count
            bib_n = len(f.get("author","").split(" and ")) if f.get("author") else 0
            cr_n = len(cr_auth)
            if bib_n and abs(bib_n - cr_n) > 0:
                issues.append(f"#AUTHORS bib={bib_n} crossref={cr_n}")
        # year
        cr_year = ""
        for kf in ("published-print","published-online","published","issued"):
            if kf in m and m[kf].get("date-parts"):
                cr_year = str(m[kf]["date-parts"][0][0]); break
        if f.get("year") and cr_year and f["year"] != cr_year:
            issues.append(f"YEAR bib={f['year']} crossref={cr_year}")
        # volume
        if f.get("volume") and m.get("volume") and f["volume"] != str(m.get("volume")):
            issues.append(f"VOLUME bib={f['volume']} crossref={m.get('volume')}")
        # issue
        if f.get("number") and m.get("issue") and f["number"] != str(m.get("issue")):
            issues.append(f"ISSUE bib={f['number']} crossref={m.get('issue')}")
        # pages
        bib_pp = clean(f.get("pages","")).replace(' ','')
        cr_pp = clean(cr_pages(m)).replace(' ','')
        if bib_pp and cr_pp and bib_pp != cr_pp:
            issues.append(f"PAGES bib={f.get('pages')} crossref={cr_pages(m)}")

        if issues:
            print(f"\n[{key}]  {doi}")
            for i in issues: print(f"    ! {i}")
            problems.append((key, issues))
        else:
            print(f"[{key}] OK")
        time.sleep(0.5)

    print("\n" + "="*72)
    print(f"DONE. {len(problems)} entries with discrepancies.")
    for k, _ in problems:
        print(f"  - {k}")

if __name__ == "__main__":
    main()
