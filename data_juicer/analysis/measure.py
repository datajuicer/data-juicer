import numpy as np

from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
td = LazyLoader("torch.distributions")
F = LazyLoader("torch.nn.functional")
stats = LazyLoader("scipy.stats")


class Measure(object):
    """Base class for Measure distribution."""

    name = "base"

    def measure(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)

    def _convert_to_tensor(self, p):
        """
        Convert input data to torch tensor.
        :param p: input data, now support
            [`scalar`,`list`, `tuple`, `torch binary file`, and `Categorical`].
        :return: torch tensor
        """
        if isinstance(p, torch.Tensor):
            return p
        elif isinstance(p, td.Categorical):
            return p.probs
        elif isinstance(p, str):
            return torch.load(p)
        else:
            return torch.tensor(p)

    def _convert_to_categorical(self, p):
        """
        Convert input data to torch Categorical.
        :param p: input data, now support
            [`scalar`,`list`, `tuple`, `torch binary file`, and `Categorical`].
        :return: torch Categorical
        """
        if isinstance(p, td.Categorical):
            return p
        elif isinstance(p, torch.Tensor):
            return td.Categorical(p)
        elif isinstance(p, str):
            return td.Categorical(torch.load(p))
        else:
            return td.Categorical(torch.tensor(p))

    def _convert_to_ndarray(self, p):
        """
        Convert input data to torch tensor.
        :param p: input data, now support
            [`scalar`,`list`, `tuple`, `torch binary file`, and `Categorical`].
        :return: torch tensor
        """
        return self._convert_to_tensor(p).numpy()


class KLDivMeasure(Measure):
    """
    Measure Kullback-Leibler divergence.
    """

    name = "kl_divergence"

    def measure(self, p, q):
        p = self._convert_to_categorical(p)
        q = self._convert_to_categorical(q)
        assert p.probs.shape == q.probs.shape, (
            "The two inputs have different shape:" f"{p.probs.shape} != {q.probs.shape} in {self.name}"
        )
        return F.kl_div(q.logits, p.probs, log_target=False, reduction="sum")


class JSDivMeasure(Measure):
    """
    Measure Jensen-Shannon divergence.
    """

    name = "js_divergence"

    def measure(self, p, q):
        p = self._convert_to_tensor(p)
        q = self._convert_to_tensor(q)
        assert p.shape == q.shape, "The two inputs have different shape:" f"{p.shape} != {q.shape} in {self.name}"

        m = 0.5 * (p + q)
        kl_p = KLDivMeasure()(p, m)
        kl_q = KLDivMeasure()(q, m)
        js = 0.5 * (kl_p + kl_q)
        return js


class CrossEntropyMeasure(Measure):
    """
    Measure Cross-Entropy.
    """

    name = "cross_entropy"

    def measure(self, p, q):
        p = self._convert_to_categorical(p)
        q = self._convert_to_categorical(q)
        assert p.probs.shape == q.probs.shape, (
            "The two inputs have different shape: " f"{p.probs.shape} != {q.probs.shape} in {self.name}"
        )
        return F.cross_entropy(q.logits, p.probs, reduction="sum")


class EntropyMeasure(Measure):
    """
    Measure Entropy.
    """

    name = "entropy"

    def measure(self, p):
        p = self._convert_to_categorical(p)
        return p.entropy()


class RelatedTTestMeasure(Measure):
    """
    Measure T-Test for two related distributions on their histogram of the same
    bins.

    Ref:
    https://en.wikipedia.org/wiki/Student%27s_t-test

    For continuous features or distributions, the input could be dataset stats
    list.
    For discrete features or distributions, the input could be the tags or the
    categories list.
    """

    name = "t-test"

    @staticmethod
    def stats_to_hist(p, q):
        p = np.array(p)
        q = np.array(q)

        # get common maximum number of data samples, and max/min values
        max_data_num = max(len(p), len(q))
        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))

        # get a recommended number of bins
        rec_bins = max(int(np.sqrt(max_data_num)), 10)

        # get the common bin edges
        common_p = np.append(p, [min_val, max_val])
        hist_p, bin_edges = np.histogram(common_p, bins=rec_bins)
        # restore the hist of the original p
        hist_p[0] -= 1
        hist_p[-1] -= 1
        # get the hist of the original q using the common bin edges
        hist_q, _ = np.histogram(q, bins=bin_edges)
        return hist_p, hist_q, bin_edges

    @staticmethod
    def category_to_hist(p, q):
        def flatten_list(lst):
            res = []
            for s in lst:
                if isinstance(s, list):
                    res.extend(flatten_list(s))
                else:
                    res.append(s)
            return res

        # flatten the list
        p = flatten_list(p)
        q = flatten_list(q)

        # get the common categories
        cat_p = set(p)
        cat_q = set(q)
        cat_common = cat_p.union(cat_q)

        # get category distributions
        count_p = {cat: 0 for cat in cat_common}
        count_q = {cat: 0 for cat in cat_common}
        for cat in p:
            count_p[cat] += 1
        for cat in q:
            count_q[cat] += 1

        # only keep distribution values sorted by counts
        sorted_cat = list(count_p.items())
        sorted_cat.sort(key=lambda it: it[1], reverse=True)
        sorted_cat = [it[0] for it in sorted_cat]
        # get the value dist
        hist_p = [count_p[cat] for cat in sorted_cat]
        hist_q = [count_q[cat] for cat in sorted_cat]

        return hist_p, hist_q, count_p, count_q, sorted_cat

    def measure(self, p, q):
        """
        :param p: the first feature or distribution. (stats/tags/categories)
        :param q: the second feature or distribution. (stats/tags/categories)
        :return: the T-Test results object -- ([ref](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.TtestResult.html#scipy.stats._result_classes.TtestResult))  # noqa: E501
        """
        ele = p[0]
        while isinstance(ele, list):
            ele = ele[0]
        if isinstance(ele, str):
            # discrete tags or categories
            hist_p, hist_q = self.category_to_hist(p, q)[:2]
        else:
            # continuous stats
            hist_p, hist_q = self.stats_to_hist(p, q)[:2]

        # compute the t-test and pval for hist_p and hist_q
        ttest_res = stats.ttest_rel(hist_p, hist_q)
        return ttest_res
