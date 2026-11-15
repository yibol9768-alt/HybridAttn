#!/usr/bin/env python3
"""Fetch metadata of arXiv ids from the arXiv API and write refs_auto.bib (no entry is written from memory)."""
import re, sys, time, urllib.request, xml.etree.ElementTree as ET

# key -> (arxiv id, venue string or None). Venues are only added for well-known publications.
IDS = {
 "vaswani2017": ("1706.03762", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "gu2024mamba": ("2312.00752", "Conference on Language Modeling (COLM)"),
 "dao2024mamba2": ("2405.21060", "International Conference on Machine Learning (ICML)"),
 "katharopoulos2020": ("2006.16236", "International Conference on Machine Learning (ICML)"),
 "schlag2021": ("2102.11174", "International Conference on Machine Learning (ICML)"),
 "yang2024deltanet": ("2406.06484", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "yang2024gla": ("2312.06635", "International Conference on Machine Learning (ICML)"),
 "yang2025gdn": ("2412.06464", "International Conference on Learning Representations (ICLR)"),
 "sun2023retnet": ("2307.08621", None),
 "peng2023rwkv": ("2305.13048", "Findings of EMNLP"),
 "peng2025rwkv7": ("2503.14456", None),
 "lieber2024jamba": ("2403.19887", "International Conference on Learning Representations (ICLR)"),
 "de2024griffin": ("2402.19427", None),
 "ren2024samba": ("2406.07522", "International Conference on Learning Representations (ICLR)"),
 "glorioso2024zamba": ("2405.16712", None),
 "blakeman2025nemotronh": ("2504.03624", None),
 "dong2024hymba": ("2411.13676", "International Conference on Learning Representations (ICLR)"),
 "qwen3": ("2505.09388", None),
 "minimax2025": ("2501.08313", None),
 "kimi2025linear": ("2510.26692", None),
 "gu2022s4": ("2111.00396", "International Conference on Learning Representations (ICLR)"),
 "gu2020hippo": ("2008.07669", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "fu2023h3": ("2212.14052", "International Conference on Learning Representations (ICLR)"),
 "arora2024zoology": ("2312.04927", "International Conference on Learning Representations (ICLR)"),
 "arora2024based": ("2402.18668", "International Conference on Machine Learning (ICML)"),
 "jelassi2024repeat": ("2402.01032", "International Conference on Machine Learning (ICML)"),
 "sun2024ttt": ("2407.04620", "International Conference on Machine Learning (ICML)"),
 "behrouz2025titans": ("2501.00663", None),
 "siems2025deltaproduct": ("2502.10297", None),
 "qin2024hgrn2": ("2404.07904", "International Conference on Machine Learning (ICML)"),
 "beck2024xlstm": ("2405.04517", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "xiao2024streamingllm": ("2309.17453", "International Conference on Learning Representations (ICLR)"),
 "zhang2023h2o": ("2306.14048", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "li2024snapkv": ("2404.14469", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "liu2023scissorhands": ("2305.17118", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "cai2024pyramidkv": ("2406.02069", None),
 "liu2024kivi": ("2402.02750", "International Conference on Machine Learning (ICML)"),
 "nawrot2024dmc": ("2403.09636", "International Conference on Machine Learning (ICML)"),
 "kwon2023vllm": ("2309.06180", "ACM Symposium on Operating Systems Principles (SOSP)"),
 "zheng2024sglang": ("2312.07104", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "gim2024promptcache": ("2311.04934", "Conference on Machine Learning and Systems (MLSys)"),
 "yao2025cacheblend": ("2405.16444", "European Conference on Computer Systems (EuroSys)"),
 "hu2024epic": ("2410.15332", None),
 "sun2024blockattention": ("2409.15355", None),
 "ye2024hydragen": ("2402.05099", None),
 "pan2025marconi": ("2411.19379", "Conference on Machine Learning and Systems (MLSys)"),
 "su2024rope": ("2104.09864", "Neurocomputing"),
 "dao2022flashattention": ("2205.14135", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "dao2024flashattention2": ("2307.08691", "International Conference on Learning Representations (ICLR)"),
 "leviathan2023spec": ("2211.17192", "International Conference on Machine Learning (ICML)"),
 "chen2023spec": ("2302.01318", None),
 "cai2024medusa": ("2401.10774", "International Conference on Machine Learning (ICML)"),
 "li2024eagle": ("2401.15077", "International Conference on Machine Learning (ICML)"),
 "bourtoule2021unlearning": ("1912.03817", "IEEE Symposium on Security and Privacy"),
 "jang2023knowledge": ("2210.01504", "Annual Meeting of the Association for Computational Linguistics (ACL)"),
 "yao2023llmunlearn": ("2310.10683", None),
 "eldan2023harry": ("2310.02238", None),
 "meng2022rome": ("2202.05262", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "meng2023memit": ("2210.07229", "International Conference on Learning Representations (ICLR)"),
 "hsieh2024ruler": ("2404.06654", "Conference on Language Modeling (COLM)"),
 "liu2024lost": ("2307.03172", "Transactions of the Association for Computational Linguistics"),
 "bai2024longbench": ("2308.14508", "Annual Meeting of the Association for Computational Linguistics (ACL)"),
 "lewis2020rag": ("2005.11401", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "packer2023memgpt": ("2310.08560", None),
 "yao2023react": ("2210.03629", "International Conference on Learning Representations (ICLR)"),
 "yao2023tot": ("2305.10601", "Advances in Neural Information Processing Systems (NeurIPS)"),
 "wang2023selfconsistency": ("2203.11171", "International Conference on Learning Representations (ICLR)"),
 "gdn2": ("2605.22791", None),
 "bole": ("2608.01651", None),
 "halo": ("2601.22156", None),
 "clm": ("2609.37725", None),
 "hyflash": ("2611.07263", None),
 "picaso": ("2502.17605", None),
 "hybridspec": ("2611.04055", None),
 "mambamod": ("2509.19633", None),
 "expressivity": ("2603.08859", None),
 "reversiblernn": ("1810.10999", "Advances in Neural Information Processing Systems (NeurIPS)"),
}

NS = {"a": "http://www.w3.org/2005/Atom"}
def fetch(ids):
    url = "https://export.arxiv.org/api/query?max_results=%d&id_list=%s" % (len(ids), ",".join(ids))
    for _ in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            return ET.fromstring(data)
        except Exception as e:
            time.sleep(3)
    raise RuntimeError("arxiv fetch failed")

def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

def main():
    items = list(IDS.items())
    got = {}
    for i in range(0, len(items), 20):
        chunk = items[i:i+20]
        root = fetch([v[0] for _, v in chunk])
        for e in root.findall("a:entry", NS):
            eid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
            eid = re.sub(r"v\d+$", "", eid)
            title = re.sub(r"\s+", " ", e.find("a:title", NS).text).strip()
            authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
            year = e.find("a:published", NS).text[:4]
            got[eid] = (title, authors, year)
        time.sleep(3)
    lines, missing = [], []
    for key, (aid, venue) in items:
        if aid not in got:
            missing.append(key); continue
        title, authors, year = got[aid]
        def fmt(n):
            p = n.split(); return p[-1] + ", " + " ".join(p[:-1]) if len(p) > 1 else n
        au = " and ".join(fmt(a) for a in authors[:12]) + (" and others" if len(authors) > 12 else "")
        entry = "@article{%s,\n  title={%s},\n  author={%s},\n  year={%s},\n" % (key, "{" + esc(title) + "}", au, year)
        entry += "  journal={arXiv preprint arXiv:%s}" % aid
        if venue:
            entry = entry.replace("journal={arXiv preprint arXiv:%s}" % aid, "journal={%s},\n  note={arXiv:%s}" % (venue, aid))
        entry += "\n}\n"
        lines.append(entry)
        print(key, "|", title[:70], "|", authors[0], "|", year)
    open("refs_auto.bib", "w").write("\n".join(lines))
    print("MISSING:", missing)

main()
