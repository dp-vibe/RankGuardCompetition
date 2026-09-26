"""Deterministic synthetic dataset generator (offline fallback).

Used when no network is available or the configured source is gated. Produces
multi-domain, varied-quality pretraining-style text in English and Chinese so
the full attack-defense pipeline can run without external data. Output is
fully deterministic given the seed.
"""

from __future__ import annotations

import random

from ..schema import Sample

_DOMAIN_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "computer_science": [
        ("en", "A sorting algorithm arranges elements in a defined order. QuickSort "
               "partitions the array around a pivot, recursing on each side; its average "
               "complexity is O(n log n), though worst-case O(n^2) occurs on already-sorted "
               "input unless the pivot is chosen randomly. In practice, introsort switches to "
               "heapsort to bound the worst case. Analysis: cs/algorithm/pivot (Knuth 1973)."),
        ("zh", "排序算法将元素按指定顺序排列。快速排序以一个基准值划分数组，对两侧递归处理，"
               "平均时间复杂度为 O(n log n)。当输入已排序时，若基准值选取不当，最坏情况退化为 O(n^2)。"
               "工程实现中常用随机化基准或切换至堆排序以保证最坏情况界。"),
    ],
    "medicine": [
        ("en", "Hypertension is a persistent elevation of arterial blood pressure. "
               "Clinical guidelines define stage 1 hypertension as systolic 130-139 mmHg or "
               "diastolic 80-89 mmHg. First-line pharmacotherapy includes thiazide diuretics, "
               "ACE inhibitors, or calcium-channel blockers; treatment is individualized to "
               "comorbidities. Trial data: ALLHAT (JAMA 2002)."),
        ("zh", "高血压是动脉血压持续升高的常见慢性病。临床指南将 1 期高血压定义为收缩压 "
               "130-139 mmHg 或舒张压 80-89 mmHg。一线药物包括噻嗪类利尿剂、ACE 抑制剂或钙通道阻滞剂，"
               "治疗方案需结合合并症个体化制定。"),
    ],
    "law": [
        ("en", "Consideration in contract law is something of value exchanged between parties. "
               "A contract lacking consideration is generally unenforceable, though promissory "
               "estoppel may substitute under the Restatement (Second) of Contracts section 90. "
               "Courts examine whether a bargained-for exchange existed."),
        ("zh", "合同法中的对价是当事人之间交换的利益。缺乏对价的合同通常不可强制执行，但在特定情形下，"
               "允诺禁反言可作为替代规则适用。法院审查是否存在协商一致的交换。"),
    ],
    "finance": [
        ("en", "The Sharpe ratio measures risk-adjusted return: (R_p - R_f) / sigma_p, where "
               "R_p is portfolio return, R_f the risk-free rate, and sigma_p the standard "
               "deviation of excess returns. Higher ratios indicate better compensation per "
               "unit of volatility."),
        ("zh", "夏普比率衡量风险调整后收益：(R_p - R_f) / sigma_p，其中 R_p 为组合收益率，"
               "R_f 为无风险利率，sigma_p 为超额收益的标准差。比率越高，单位波动获得的补偿越好。"),
    ],
    "history": [
        ("en", "The printing press, introduced in Europe by Gutenberg around 1440, sharply "
               "reduced the cost of reproducing texts and is widely credited with accelerating "
               "the spread of literacy and the Protestant Reformation in the 16th century."),
        ("zh", "古登堡于 1440 年前后在欧洲引入活字印刷术，大幅降低了书籍复制成本，"
               "被广泛认为加速了识字率的普及与 16 世纪宗教改革的发展。"),
    ],
    "literature": [
        ("en", "In free indirect discourse, the narrator relays a character's thoughts without "
               "quotation marks, blending the character's voice with the narrator's. Austen and "
               "Flaubert used the technique to render interiority while preserving ironic distance."),
        ("zh", "自由间接引语由叙述者转述人物内心活动，不加引号，使人物声音与叙述者声音交融。"
               "简·奥斯汀与福楼拜善用此技法，在呈现内心活动的同时保持反讽距离。"),
    ],
    "mathematics": [
        ("en", "A group is a set G with a binary operation satisfying closure, associativity, "
               "an identity element, and inverses. Lagrange's theorem states that for any finite "
               "group G and subgroup H, the order of H divides the order of G."),
        ("zh", "群是一个集合 G 配合一个二元运算，满足封闭性、结合律、存在单位元与逆元。"
               "拉格朗日定理指出：对有限群 G 及其子群 H，H 的阶整除 G 的阶。"),
    ],
    "physics": [
        ("en", "The Heisenberg uncertainty principle states that position x and momentum p "
               "cannot both be known precisely: Delta x * Delta p >= hbar/2. It follows from "
               "the non-commutativity of the position and momentum operators in quantum mechanics."),
        ("zh", "海森堡不确定性原理指出，位置 x 与动量 p 不能同时被精确测定：Delta x * Delta p >= hbar/2。"
               "该结论源于量子力学中位置算符与动量算符的不对易性。"),
    ],
}

# Low-quality filler to drag some samples down the ranking.
_LOW_QUALITY = [
    "blah blah blah stuff stuff more stuff lol click here now!!!",
    "todo write this section later placeholder random words keyword stuff.",
    "buy now cheap deal amazing offer click link subscribe today wow",
    "this page is empty. nothing here. come back later maybe. ads ads ads.",
]


def _make_text(rnd: random.Random, domain: str, lang: str, quality: str) -> str:
    tmpl = next((t for l, t in _DOMAIN_TEMPLATES[domain] if l == lang),
                _DOMAIN_TEMPLATES[domain][0])
    if quality == "low":
        # corrupt: truncate and append filler
        cut = len(tmpl) // 3
        return tmpl[:cut] + " " + rnd.choice(_LOW_QUALITY)
    if quality == "medium":
        # drop a sentence boundary and lightly garble
        parts = tmpl.split(". ")
        rnd.shuffle(parts)
        return ". ".join(parts[:2]) + "."
    return tmpl  # high


def generate_samples(
    n: int,
    split: str,
    domains: list[str],
    languages: list[str],
    seed: int = 42,
    start_index: int = 0,
) -> list[Sample]:
    """Produce ``n`` deterministic synthetic samples for a split."""
    rnd = random.Random(f"{seed}-{split}-{start_index}")
    samples: list[Sample] = []
    # quality distribution: 50% high, 30% medium, 20% low -> non-trivial ranking
    qualities = (["high"] * 5 + ["medium"] * 3 + ["low"] * 2)
    for i in range(n):
        domain = domains[(i + start_index) % len(domains)]
        lang = languages[i % len(languages)]
        if domain not in _DOMAIN_TEMPLATES:
            domain = "computer_science"
        quality = qualities[(i + start_index) % len(qualities)]
        text = _make_text(rnd, domain, lang, quality)
        sid = f"{split}_{domain[:4]}_{i:04d}"
        samples.append(Sample(
            sample_id=sid,
            domain=domain,
            language=lang,
            clean_text=text,
            inject_position="append",
        ))
    return samples
