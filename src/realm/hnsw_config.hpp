/*************************************************************************
 *
 * Copyright 2025 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#ifndef REALM_HNSW_CONFIG_HPP
#define REALM_HNSW_CONFIG_HPP

#include <cstddef>

namespace realm {

// Distance metric types for HNSW vector search
enum class DistanceMetric {
    Euclidean,   // L2 distance: sqrt(sum((v1[i] - v2[i])^2))
    Cosine,      // Cosine distance: 1 - (dot / (norm1 * norm2))
    DotProduct   // Negative dot product for maximum inner product search
};

// Configuration parameters for HNSW index
// This is a lightweight struct with no dependencies, safe to include in table.hpp
struct HNSWIndexConfig {
    DistanceMetric metric;
    size_t M;
    size_t M0;
    size_t ef_construction;
    size_t ef_search;
    double ml;
    size_t vector_dimension;
    uint64_t random_seed;
    
    // Constructor requires explicit metric choice
    explicit HNSWIndexConfig(DistanceMetric dist_metric)
        : metric(dist_metric)
        , M(16)
        , M0(32)
        , ef_construction(200)
        , ef_search(50)
        , ml(1.0 / 1.442695040888963) // 1.0 / log(2.0)
        , vector_dimension(0)
        , random_seed(42)
    {}
};

} // namespace realm

#endif // REALM_HNSW_CONFIG_HPP
