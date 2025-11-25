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

#ifndef REALM_INDEX_HNSW_HPP
#define REALM_INDEX_HNSW_HPP

#include <realm/array.hpp>
#include <realm/search_index.hpp>
#include <realm/list.hpp>
#include <realm/hnsw_config.hpp>
#include <vector>
#include <unordered_map>
#include <queue>
#include <random>
#include <cmath>
#include <limits>
#include <shared_mutex>
#include <atomic>

namespace realm {

/**
 * HNSW (Hierarchical Navigable Small World) Index for vector similarity search.
 * 
 * This implements an efficient approximate nearest neighbor search for List<double> vectors.
 * The algorithm builds a multi-layer graph where:
 * - Higher layers have sparse connections for long-range navigation
 * - Lower layers have dense connections for fine-grained search
 * - Layer 0 contains all points, upper layers contain exponentially fewer points
 * 
 * Key features:
 * - Sub-linear search time complexity: O(log N) for most queries
 * - No need to scan all records in the database
 * - Supports incremental insertion of vectors
 * - Tunable parameters for accuracy/speed tradeoff
 */

class HNSWIndex : public SearchIndex {
public:
    // HNSW Configuration parameters (using shared HNSWIndexConfig)
    using Config = HNSWIndexConfig;

    HNSWIndex(const ClusterColumn& target_column, Allocator& alloc, const Config& config);
    HNSWIndex(ref_type ref, ArrayParent* parent, size_t ndx_in_parent, 
              const ClusterColumn& target_column, Allocator& alloc, const Config& config = Config(DistanceMetric::Euclidean));
    ~HNSWIndex() override;

    // SearchIndex interface implementation
    void insert(ObjKey key, const Mixed& value) override;
    void set(ObjKey key, const Mixed& value) override;
    void erase(ObjKey key) override;
    ObjKey find_first(const Mixed& value) const override;
    void find_all(std::vector<ObjKey>& result, Mixed value, bool case_insensitive = false) const override;
    FindRes find_all_no_copy(Mixed value, InternalFindResult& result) const override;
    size_t count(const Mixed& value) const override;
    void clear() override;
    bool has_duplicate_values() const noexcept override { return false; }
    bool is_empty() const override;
    void insert_bulk(const ArrayUnsigned* keys, uint64_t key_offset, size_t num_values,
                     ArrayPayload& values) override;
    void insert_bulk_list(const ArrayUnsigned* keys, uint64_t key_offset, size_t num_values,
                          ArrayInteger& ref_array) override;
    void verify() const override;
#ifdef REALM_DEBUG
    void print() const override;
#endif

    // Vector similarity search API
    /**
     * Search for k nearest neighbors to the query vector
     * @param query_vector The vector to search for
     * @param k Number of nearest neighbors to find
     * @param ef_search Size of dynamic candidate list (overrides config if > 0)
     * @return Vector of (ObjKey, distance) pairs sorted by distance (ascending)
     */
    std::vector<std::pair<ObjKey, double>> search_knn(const std::vector<double>& query_vector, 
                                                        size_t k, 
                                                        size_t ef_search = 0) const;

    /**
     * Search for all vectors within a given distance threshold
     * @param query_vector The vector to search for
     * @param max_distance Maximum distance threshold
     * @return Vector of (ObjKey, distance) pairs for all vectors within threshold
     */
    std::vector<std::pair<ObjKey, double>> search_radius(const std::vector<double>& query_vector,
                                                          double max_distance) const;

    // Configuration
    const Config& get_config() const { return m_config; }
    void set_ef_search(size_t ef_search) { m_config.ef_search = ef_search; }
    ObjKey get_entry_point() const { return m_entry_point; }
    // Rebuild the graph (simple brute force reinsertion). Expensive; use sparingly.
    void rebuild();
    
    // Statistics
    size_t get_num_vectors() const { return m_vectors.size(); }
    int get_max_layer() const { return m_entry_point_layer; }

private:
    // Internal data structures
    struct Node {
        ObjKey obj_key;
        std::vector<double> vector;
        int layer;  // Highest layer this node appears in
        std::vector<std::vector<ObjKey>> connections;  // connections[layer] = list of connected nodes
    };

    // Priority queue element for search
    struct SearchCandidate {
        ObjKey obj_key;
        double distance;
        
        bool operator<(const SearchCandidate& other) const {
            return distance < other.distance;  // Min-heap
        }
        
        bool operator>(const SearchCandidate& other) const {
            return distance > other.distance;  // Max-heap
        }
    };

    Config m_config;
    std::unique_ptr<Array> m_array;  // Root array for persistence
    
    // In-memory index structures
    std::unordered_map<int64_t, Node> m_vectors;  // Main vector storage (keyed by ObjKey::value)
    ObjKey m_entry_point;                                     // Entry point for search (highest layer node)
    int m_entry_point_layer;                                  // Layer of entry point
    mutable std::mt19937_64 m_rng;                           // Random number generator
    mutable std::shared_mutex m_mutex;                       // Concurrency control (read: shared, write: unique)
    static constexpr uint64_t k_format_version = 1;          // Persistence format version

    // Metrics counters
    struct Metrics {
        std::atomic<uint64_t> insert_count{0};
        std::atomic<uint64_t> erase_count{0};
        std::atomic<uint64_t> search_count{0};
        std::atomic<uint64_t> radius_search_count{0};
        std::atomic<uint64_t> total_insert_ns{0};
        std::atomic<uint64_t> total_search_ns{0};
        std::atomic<uint64_t> total_radius_search_ns{0};
    } mutable m_metrics;

public: // metrics accessors
    uint64_t get_insert_count() const { return m_metrics.insert_count.load(); }
    uint64_t get_search_count() const { return m_metrics.search_count.load(); }
    uint64_t get_radius_search_count() const { return m_metrics.radius_search_count.load(); }
    double get_avg_insert_ms() const {
        uint64_t c = m_metrics.insert_count.load();
        return c ? (m_metrics.total_insert_ns.load() / 1e6) / c : 0.0;
    }
    double get_avg_search_ms() const {
        uint64_t c = m_metrics.search_count.load();
        return c ? (m_metrics.total_search_ns.load() / 1e6) / c : 0.0;
    }
    double get_avg_radius_search_ms() const {
        uint64_t c = m_metrics.radius_search_count.load();
        return c ? (m_metrics.total_radius_search_ns.load() / 1e6) / c : 0.0;
    }

    // Distance computation
    double compute_distance(const std::vector<double>& v1, const std::vector<double>& v2) const;
    double euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2) const;
    double cosine_distance(const std::vector<double>& v1, const std::vector<double>& v2) const;
    double dot_product_distance(const std::vector<double>& v1, const std::vector<double>& v2) const;
    
    // Vector extraction from Mixed/List<double>
    std::vector<double> extract_vector(const Mixed& value) const;
    std::vector<double> get_vector_for_key(ObjKey key) const;
    
    // HNSW algorithm internals
    int select_layer();
    std::vector<ObjKey> search_layer(const std::vector<double>& query, ObjKey entry_point, 
                                      size_t ef, int layer) const;
    std::vector<std::pair<ObjKey, double>> search_layer_with_distances(
        const std::vector<double>& query, ObjKey entry_point, size_t ef, int layer) const;
    std::vector<ObjKey> select_neighbors_simple(const std::vector<double>& query,
                                                 const std::vector<std::pair<ObjKey, double>>& candidates,
                                                 size_t M) const;
    std::vector<ObjKey> select_neighbors_heuristic(const std::vector<double>& query,
                                                    const std::vector<std::pair<ObjKey, double>>& candidates,
                                                    size_t M, int layer, bool extend_candidates = false) const;
    void prune_connections(ObjKey node_key, int layer);
    
    // Graph maintenance
    void connect_nodes(ObjKey node1, ObjKey node2, int layer);
    void disconnect_nodes(ObjKey node1, ObjKey node2, int layer);
    
    // Persistence helpers
    void load_from_storage();
    void save_to_storage();
    
    // Validation
    void validate_vector_dimension(const std::vector<double>& vector) const;
};

} // namespace realm

#endif // REALM_INDEX_HNSW_HPP
