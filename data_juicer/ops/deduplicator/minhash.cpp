#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

// Constants
const uint32_t MERSENNE_PRIME = 2147483647;  // 2^31 - 1
const uint32_t MAX_HASH = 4294967295;  // 2^32 - 1

uint32_t simple_hash(const std::string& token) {
    uint32_t hash = 5381;
    for (const uint8_t c : token) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

std::vector<std::tuple<uint32_t, py::bytes, uint64_t>> calc_minhash_c(
    const std::vector<std::string>& tokens,
    const py::array_t<uint32_t>& perm_a,
    const py::array_t<uint32_t>& perm_b,
    const py::bytes& empty_hash_value,
    const std::vector<std::pair<size_t, size_t>>& hash_ranges,
    uint32_t union_find_parallel_num,
    uint64_t uid)
{
    std::vector<std::tuple<uint32_t, py::bytes, uint64_t>> pairs;

    if (tokens.empty()) {
        pairs.emplace_back(MAX_HASH % union_find_parallel_num, empty_hash_value, uid);
        return pairs;
    }

    std::vector<uint32_t> hv;
    hv.reserve(tokens.size());
    for (const std::string& token : tokens) {
        hv.push_back(simple_hash(token));
    }

    auto perm_a_data = perm_a.unchecked<1>();
    auto perm_b_data = perm_b.unchecked<1>();
    size_t num_permutation = perm_a.shape(0);

    std::vector<uint32_t> hash_values(num_permutation, MAX_HASH);
    for (size_t i = 0; i < num_permutation; ++i) {
        for (uint32_t h : hv) {
            uint32_t phv = ((static_cast<uint64_t>(h) * perm_a_data(i) + perm_b_data(i)) % MERSENNE_PRIME) & MAX_HASH;
            hash_values[i] = std::min(hash_values[i], phv);
        }
    }


    for (size_t i = 0; i < hash_ranges.size(); ++i) {
        const auto& [start, end] = hash_ranges[i];
        std::vector<uint32_t> band_hash_values(hash_values.begin() + start, hash_values.begin() + end);

        py::bytes hash_value = py::bytes(
            std::string(reinterpret_cast<char*>(&i), sizeof(uint32_t)) +
            std::string(reinterpret_cast<char*>(band_hash_values.data()), band_hash_values.size() * sizeof(uint32_t))
        );

        uint32_t hash_table_id = hash_values[start] % union_find_parallel_num;
        pairs.emplace_back(hash_table_id, hash_value, uid);
    }

    return pairs;
}

py::list calc_minhash_batch_c(
    const std::vector<std::vector<std::string>>& tokens_list,
    const uint64_t uid_begin,
    const std::vector<uint64_t>& perm_a,
    const std::vector<uint64_t>& perm_b,
    const std::string& empty_hash_value,
    const std::vector<std::pair<size_t, size_t>>& hash_ranges,
    uint32_t union_find_parallel_num,
    uint32_t num_threads)
{
    omp_set_num_threads(num_threads);
    size_t total_docs = tokens_list.size();
    std::vector<std::tuple<uint32_t, std::string, uint64_t>> intermediate_pairs;
    intermediate_pairs.reserve(total_docs * hash_ranges.size());

    size_t num_permutation = perm_a.size();

    #pragma omp parallel
    {
        std::vector<std::tuple<uint32_t, std::string, uint64_t>> local_pairs;
        local_pairs.reserve(total_docs * hash_ranges.size() / num_threads);
        std::vector<uint32_t> hash_values(num_permutation);

        #pragma omp for nowait
        for (size_t doc_idx = 0; doc_idx < total_docs; ++doc_idx) {
            const auto& tokens = tokens_list[doc_idx];
            uint64_t uid = uid_begin + doc_idx;

            if (tokens.empty()) {
                local_pairs.emplace_back(MAX_HASH % union_find_parallel_num, empty_hash_value, uid);
                continue;
            }

            std::fill(hash_values.begin(), hash_values.end(), MAX_HASH);
            for (const auto& token : tokens) {
                uint32_t h = simple_hash(token);
                for (size_t i = 0; i < num_permutation; ++i) {
                    uint32_t phv = (static_cast<uint64_t>(h) * perm_a[i] + perm_b[i]) >> 32;
                    hash_values[i] = std::min(hash_values[i], phv);
                }
            }

            for (size_t i = 0; i < hash_ranges.size(); ++i) {
                const auto& [start, end] = hash_ranges[i];
                std::string hash_value(reinterpret_cast<char*>(&i), sizeof(uint32_t));
                hash_value.append(reinterpret_cast<char*>(&hash_values[start]), (end - start) * sizeof(uint32_t));

                uint32_t hash_table_id = hash_values[start] % union_find_parallel_num;
                local_pairs.emplace_back(hash_table_id, std::move(hash_value), uid);
            }
        }

        #pragma omp critical
        {
            intermediate_pairs.insert(intermediate_pairs.end(), local_pairs.begin(), local_pairs.end());
        }
    }
    py::list result;
    for (const auto& item : intermediate_pairs) {
        uint32_t first = std::get<0>(item);
        py::bytes second = py::bytes(std::get<1>(item));
        uint64_t third = std::get<2>(item);
        result.append(py::make_tuple(first, second, third));
    }
    return result;
}

std::vector<std::tuple<uint32_t, py::bytes>> calc_simple_minhash_c(
    const std::vector<std::string>& tokens,
    const py::array_t<uint32_t>& perm_a,
    const py::array_t<uint32_t>& perm_b,
    const std::vector<std::pair<size_t, size_t>>& hash_ranges,
    uint32_t bucket_per_band,
    uint64_t uid)
{
    std::vector<std::tuple<uint32_t, py::bytes>> pairs;

    if (tokens.empty()) {
        pairs.emplace_back(0, py::bytes(""));
        return pairs;
    }

    std::vector<uint32_t> hv;
    hv.reserve(tokens.size());
    for (const std::string& token : tokens) {
        hv.push_back(simple_hash(token));
    }

    auto perm_a_data = perm_a.unchecked<1>();
    auto perm_b_data = perm_b.unchecked<1>();
    size_t num_permutation = perm_a.shape(0);

    std::vector<uint32_t> hash_values(num_permutation, MAX_HASH);
    for (size_t i = 0; i < num_permutation; ++i) {
        for (uint32_t h : hv) {
            uint32_t phv = ((static_cast<uint64_t>(h) * perm_a_data(i) + perm_b_data(i)) % MERSENNE_PRIME) & MAX_HASH;
            hash_values[i] = std::min(hash_values[i], phv);
        }
    }


    for (size_t i = 0; i < hash_ranges.size(); ++i) {
        const auto& [start, end] = hash_ranges[i];
        std::vector<uint32_t> band_hash_values(hash_values.begin() + start, hash_values.begin() + end);

        py::bytes hash_value = py::bytes(
            std::string(reinterpret_cast<char*>(band_hash_values.data()), band_hash_values.size() * sizeof(uint32_t))
        );

        uint32_t hash_table_id = bucket_per_band * i + (hash_values[start] % bucket_per_band);
        pairs.emplace_back(hash_table_id, hash_value);
    }

    return pairs;
}


PYBIND11_MODULE(minhash, m) {
    m.def("calc_minhash_c", &calc_minhash_c, "C++ implementation of calc_minhash");
    m.def("calc_simple_minhash_c", &calc_simple_minhash_c, "C++ implementation of calc_simple_minhash");
    m.def("calc_minhash_batch_c", &calc_minhash_batch_c, "C++ implementation of calc_minhash (batch version)");
}
