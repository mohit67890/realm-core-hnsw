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

#include <realm/index_hnsw.hpp>
#include <realm/array_integer.hpp>
#include <realm/column_integer.hpp>
#include <realm/exceptions.hpp>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <shared_mutex>
#include <chrono>

namespace realm {

// ===================== Constructor / Destructor =====================

HNSWIndex::HNSWIndex(const ClusterColumn& target_column, Allocator& alloc, const Config& config)
    : SearchIndex(target_column, nullptr)
    , m_config(config)
    , m_array(std::make_unique<Array>(alloc))
    , m_entry_point()
    , m_entry_point_layer(-1)
    , m_rng(config.random_seed)
{
    m_array->create(Array::type_HasRefs);
    m_root_array = m_array.get();
    // Normalize configuration defaults
    if (m_config.M0 == 0) {
        m_config.M0 = m_config.M * 2; // typical heuristic
    }
    if (m_config.ef_search == 0) {
        m_config.ef_search = std::max<size_t>(64, m_config.M * 8);
    }
}

HNSWIndex::HNSWIndex(ref_type ref, ArrayParent* parent, size_t ndx_in_parent,
                     const ClusterColumn& target_column, Allocator& alloc, const Config& config)
    : SearchIndex(target_column, nullptr)
    , m_config(config)
    , m_array(std::make_unique<Array>(alloc))
    , m_entry_point()
    , m_entry_point_layer(-1)
    , m_rng(config.random_seed)
{
    m_array->init_from_ref(ref);
    m_array->set_parent(parent, ndx_in_parent);
    m_root_array = m_array.get();
    load_from_storage();
    if (m_config.M0 == 0) {
        m_config.M0 = m_config.M * 2;
    }
    if (m_config.ef_search == 0) {
        m_config.ef_search = std::max<size_t>(64, m_config.M * 8);
    }
}

HNSWIndex::~HNSWIndex() = default;

// ===================== Distance Metrics =====================

double HNSWIndex::euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2) const
{
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double HNSWIndex::cosine_distance(const std::vector<double>& v1, const std::vector<double>& v2) const
{
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 1.0;  // Maximum distance for zero vectors
    }
    double cosine_sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
    return 1.0 - cosine_sim;  // Convert similarity to distance
}

double HNSWIndex::dot_product_distance(const std::vector<double>& v1, const std::vector<double>& v2) const
{
    double dot = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
    }
    return -dot;  // Negative for maximum inner product search
}

double HNSWIndex::compute_distance(const std::vector<double>& v1, const std::vector<double>& v2) const
{
    switch (m_config.metric) {
        case DistanceMetric::Euclidean:
            return euclidean_distance(v1, v2);
        case DistanceMetric::Cosine:
            return cosine_distance(v1, v2);
        case DistanceMetric::DotProduct:
            return dot_product_distance(v1, v2);
        default:
            REALM_UNREACHABLE();
    }
}

// ===================== Vector Extraction =====================

std::vector<double> HNSWIndex::extract_vector(const Mixed& value) const
{
    std::vector<double> result;
    
    // Handle direct vector data (if Mixed contains a list reference)
    if (value.is_type(type_TypedLink)) {
        // Get the actual list from the object
        ObjLink link = value.get<ObjLink>();
        ObjKey key = link.get_obj_key();
        return get_vector_for_key(key);
    }
    
    return result;
}

std::vector<double> HNSWIndex::get_vector_for_key(ObjKey key) const
{
    std::vector<double> result;
    
    // Get the table and object
    const ClusterTree* cluster_tree = m_target_column.get_cluster_tree();
    if (!cluster_tree) {
        return result;
    }
    
    // Get the object directly from cluster tree
    const Obj obj{cluster_tree->get(key)};
    ColKey col_key = m_target_column.get_column_key();
    
    // Get the list of doubles
    Lst<double> list = obj.get_list<double>(col_key);
    result.reserve(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        result.push_back(list.get(i));
    }
    
    return result;
}

void HNSWIndex::validate_vector_dimension(const std::vector<double>& vector) const
{
    if (m_config.vector_dimension == 0) {
        // First vector - set dimension
        const_cast<Config&>(m_config).vector_dimension = vector.size();
    } else if (vector.size() != m_config.vector_dimension) {
        throw InvalidArgument(ErrorCodes::InvalidQuery,
            "Vector dimension mismatch: expected " + std::to_string(m_config.vector_dimension) +
            " but got " + std::to_string(vector.size()));
    }
}

// ===================== Layer Selection =====================

int HNSWIndex::select_layer()
{
    // Select layer with exponential decay probability
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double r = uniform(m_rng);
    int layer = static_cast<int>(-log(r) * m_config.ml);
    constexpr int k_max_layer_cap = 32;
    if (layer > k_max_layer_cap) layer = k_max_layer_cap;
    return layer;
}

// ===================== Search Layer Algorithm =====================

std::vector<std::pair<ObjKey, double>> HNSWIndex::search_layer_with_distances(
    const std::vector<double>& query, ObjKey entry_point, size_t ef, int layer) const
{
    std::vector<std::pair<ObjKey, double>> result;
    
    if (!entry_point || m_vectors.empty()) {
        return result;
    }
    
    // Priority queues for candidates and visited set
    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::greater<SearchCandidate>> candidates;  // Min-heap
    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::less<SearchCandidate>> w;  // Max-heap for top-ef
    std::unordered_set<int64_t> visited;
    
    // Initialize with entry point
    auto entry_it = m_vectors.find(entry_point.value);
    if (entry_it == m_vectors.end()) {
        return result;
    }
    
    double entry_dist = compute_distance(query, entry_it->second.vector);
    candidates.push({entry_point, entry_dist});
    w.push({entry_point, entry_dist});
    visited.insert(entry_point.value);
    
    // Greedy search
    while (!candidates.empty()) {
        SearchCandidate current = candidates.top();
        candidates.pop();
        
        // If current distance is worse than worst in result set, stop
        if (current.distance > w.top().distance && w.size() >= ef) {
            // We have gathered at least ef candidates and the best remaining candidate
            // is worse than the worst accepted -> terminate per HNSW logic
            break;
        }
        
        // Check all neighbors at this layer
        auto node_it = m_vectors.find(current.obj_key.value);
        if (node_it == m_vectors.end()) {
            continue;
        }
        
        const Node& node = node_it->second;
        if (layer >= static_cast<int>(node.connections.size())) {
            continue;
        }
        
        for (ObjKey neighbor_key : node.connections[layer]) {
            if (visited.find(neighbor_key.value) != visited.end()) {
                continue;
            }
            visited.insert(neighbor_key.value);
            
            auto neighbor_it = m_vectors.find(neighbor_key.value);
            if (neighbor_it == m_vectors.end()) {
                continue;
            }
            
            double neighbor_dist = compute_distance(query, neighbor_it->second.vector);
            
            if (neighbor_dist < w.top().distance || w.size() < ef) {
                candidates.push({neighbor_key, neighbor_dist});
                w.push({neighbor_key, neighbor_dist});
                
                if (w.size() > ef) {
                    w.pop();
                }
            }
        }
    }
    
    // Extract results from max-heap
    while (!w.empty()) {
        result.push_back({w.top().obj_key, w.top().distance});
        w.pop();
    }
    
    // Reverse to get ascending order
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<ObjKey> HNSWIndex::search_layer(const std::vector<double>& query, ObjKey entry_point,
                                             size_t ef, int layer) const
{
    auto results_with_dist = search_layer_with_distances(query, entry_point, ef, layer);
    std::vector<ObjKey> results;
    results.reserve(results_with_dist.size());
    for (const auto& pair : results_with_dist) {
        results.push_back(pair.first);
    }
    return results;
}

// ===================== Neighbor Selection =====================

std::vector<ObjKey> HNSWIndex::select_neighbors_simple(
    const std::vector<double>& query,
    const std::vector<std::pair<ObjKey, double>>& candidates,
    size_t M) const
{
    std::vector<ObjKey> result;
    size_t count = std::min(M, candidates.size());
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(candidates[i].first);
    }
    return result;
}

std::vector<ObjKey> HNSWIndex::select_neighbors_heuristic(
    const std::vector<double>& query,
    const std::vector<std::pair<ObjKey, double>>& candidates,
    size_t M, int layer, bool extend_candidates) const
{
    // Heuristic neighbor selection to maintain graph quality
    std::vector<std::pair<ObjKey, double>> working_set = candidates;
    std::vector<ObjKey> result;
    result.reserve(M);
    
    if (extend_candidates) {
        // Add neighbors of candidates to working set
        std::unordered_set<int64_t> in_working_set;
        for (const auto& c : candidates) {
            in_working_set.insert(c.first.value);
        }
        
        for (const auto& c : candidates) {
            auto node_it = m_vectors.find(c.first.value);
            if (node_it != m_vectors.end() && layer < static_cast<int>(node_it->second.connections.size())) {
                for (ObjKey neighbor : node_it->second.connections[layer]) {
                    if (in_working_set.find(neighbor.value) == in_working_set.end()) {
                        auto neighbor_it = m_vectors.find(neighbor.value);
                        if (neighbor_it != m_vectors.end()) {
                            double dist = compute_distance(query, neighbor_it->second.vector);
                            working_set.push_back({neighbor, dist});
                            in_working_set.insert(neighbor.value);
                        }
                    }
                }
            }
        }
        
        std::sort(working_set.begin(), working_set.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
    }
    
    // Select diverse neighbors
    for (const auto& candidate : working_set) {
        if (result.size() >= M) {
            break;
        }
        
        // Check if candidate is closer to query than to any already selected neighbor
        bool should_add = true;
        auto cand_it = m_vectors.find(candidate.first.value);
        if (cand_it != m_vectors.end()) {
            for (ObjKey selected : result) {
                auto sel_it = m_vectors.find(selected.value);
                if (sel_it != m_vectors.end()) {
                    double dist_to_selected = compute_distance(cand_it->second.vector, sel_it->second.vector);
                    if (dist_to_selected < candidate.second) {
                        should_add = false;
                        break;
                    }
                }
            }
        }
        
        if (should_add) {
            result.push_back(candidate.first);
        }
    }
    
    return result;
}

// ===================== Graph Operations =====================

void HNSWIndex::connect_nodes(ObjKey node1, ObjKey node2, int layer)
{
    auto it1 = m_vectors.find(node1.value);
    auto it2 = m_vectors.find(node2.value);
    
    if (it1 == m_vectors.end() || it2 == m_vectors.end()) {
        return;
    }
    
    Node& n1 = it1->second;
    Node& n2 = it2->second;
    
    // Ensure connections vector is large enough
    while (static_cast<int>(n1.connections.size()) <= layer) {
        n1.connections.push_back({});
    }
    while (static_cast<int>(n2.connections.size()) <= layer) {
        n2.connections.push_back({});
    }
    
    // Add bidirectional edges
    auto& conn1 = n1.connections[layer];
    auto& conn2 = n2.connections[layer];
    
    if (std::find(conn1.begin(), conn1.end(), node2) == conn1.end()) {
        conn1.push_back(node2);
    }
    if (std::find(conn2.begin(), conn2.end(), node1) == conn2.end()) {
        conn2.push_back(node1);
    }
}

void HNSWIndex::disconnect_nodes(ObjKey node1, ObjKey node2, int layer)
{
    auto it1 = m_vectors.find(node1.value);
    auto it2 = m_vectors.find(node2.value);
    
    if (it1 == m_vectors.end() || it2 == m_vectors.end()) {
        return;
    }
    
    Node& n1 = it1->second;
    Node& n2 = it2->second;
    
    if (layer < static_cast<int>(n1.connections.size())) {
        auto& conn1 = n1.connections[layer];
        conn1.erase(std::remove(conn1.begin(), conn1.end(), node2), conn1.end());
    }
    
    if (layer < static_cast<int>(n2.connections.size())) {
        auto& conn2 = n2.connections[layer];
        conn2.erase(std::remove(conn2.begin(), conn2.end(), node1), conn2.end());
    }
}

void HNSWIndex::prune_connections(ObjKey node_key, int layer)
{
    auto node_it = m_vectors.find(node_key.value);
    if (node_it == m_vectors.end() || layer >= static_cast<int>(node_it->second.connections.size())) {
        return;
    }
    
    Node& node = node_it->second;
    size_t max_conn = (layer == 0) ? m_config.M0 : m_config.M;
    
    if (node.connections[layer].size() <= max_conn) {
        return;
    }
    
    // Build candidate list with distances
    std::vector<std::pair<ObjKey, double>> candidates;
    for (ObjKey neighbor : node.connections[layer]) {
        auto neighbor_it = m_vectors.find(neighbor.value);
        if (neighbor_it != m_vectors.end()) {
            double dist = compute_distance(node.vector, neighbor_it->second.vector);
            candidates.push_back({neighbor, dist});
        }
    }
    
    // Sort by distance
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Select best neighbors
    std::vector<ObjKey> new_neighbors = select_neighbors_heuristic(node.vector, candidates, max_conn, layer, false);
    
    // Remove old connections
    for (ObjKey old_neighbor : node.connections[layer]) {
        if (std::find(new_neighbors.begin(), new_neighbors.end(), old_neighbor) == new_neighbors.end()) {
            disconnect_nodes(node_key, old_neighbor, layer);
        }
    }
    
    // Update connections
    node.connections[layer] = new_neighbors;
}

// ===================== Insert Operation =====================

void HNSWIndex::insert(ObjKey key, const Mixed& value)
{
    std::unique_lock lock(m_mutex);
    auto t0 = std::chrono::high_resolution_clock::now();
    // Extract vector from the list column
    std::vector<double> vector = get_vector_for_key(key);
    
    if (vector.empty()) {
        return;  // No vector to index
    }
    
    validate_vector_dimension(vector);
    
    // Determine layer for new node
    int node_layer = select_layer();
    
    // Create new node
    Node new_node;
    new_node.obj_key = key;
    new_node.vector = std::move(vector);
    new_node.layer = node_layer;
    new_node.connections.resize(node_layer + 1);
    
    // If this is the first node, make it the entry point
    if (m_vectors.empty()) {
        m_vectors[key.value] = std::move(new_node);
        m_entry_point = key;
        m_entry_point_layer = node_layer;
        save_to_storage();
        return;
    }
    
    // Search for nearest neighbors
    std::vector<ObjKey> nearest;
    ObjKey curr_nearest = m_entry_point;
    
    // Traverse from top layer to target layer
    for (int lc = m_entry_point_layer; lc > node_layer; --lc) {
        auto results = search_layer(new_node.vector, curr_nearest, 1, lc);
        if (!results.empty()) {
            curr_nearest = results[0];
        }
    }
    
    // Insert node and connect at each layer from node_layer down to 0
    // Add node to index once before establishing connections
    m_vectors[key.value] = new_node;
    for (int lc = node_layer; lc >= 0; --lc) {
        size_t ef = m_config.ef_construction;
        auto candidates_with_dist = search_layer_with_distances(new_node.vector, curr_nearest, ef, lc);
        
        size_t M = (lc == 0) ? m_config.M0 : m_config.M;
        std::vector<ObjKey> neighbors;
        if (lc == 0) {
            neighbors = select_neighbors_simple(new_node.vector, candidates_with_dist, M);
        } else {
            neighbors = select_neighbors_heuristic(new_node.vector, candidates_with_dist, M, lc, true);
        }
        
        // Connect new node to neighbors
        for (ObjKey neighbor : neighbors) {
            connect_nodes(key, neighbor, lc);
        }
        
        // Prune neighbors' connections if needed
        for (ObjKey neighbor : neighbors) {
            prune_connections(neighbor, lc);
        }
        
        if (!candidates_with_dist.empty()) {
            curr_nearest = candidates_with_dist[0].first;
        }
    }
    
    // Update entry point if new node is on a higher layer
    if (node_layer > m_entry_point_layer) {
        m_entry_point = key;
        m_entry_point_layer = node_layer;
    }
    
    // Persist changes to storage
    save_to_storage();
    auto t1 = std::chrono::high_resolution_clock::now();
    m_metrics.insert_count.fetch_add(1);
    m_metrics.total_insert_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

// ===================== Other SearchIndex Operations =====================

void HNSWIndex::set(ObjKey key, const Mixed& value)
{
    // Note: erase() and insert() each handle their own locking
    // Remove old entry if exists
    erase(key);
    // Insert new entry (insert will call save_to_storage)
    insert(key, value);
}

void HNSWIndex::erase(ObjKey key)
{
    std::unique_lock lock(m_mutex);
    auto it = m_vectors.find(key.value);
    if (it == m_vectors.end()) {
        return;
    }
    
    Node& node = it->second;
    
    // Disconnect from all neighbors at all layers
    for (int layer = 0; layer <= node.layer; ++layer) {
        if (layer < static_cast<int>(node.connections.size())) {
            for (ObjKey neighbor : node.connections[layer]) {
                disconnect_nodes(key, neighbor, layer);
            }
        }
    }
    
    // Remove node
    m_vectors.erase(it);
    
    // Update entry point if necessary
    if (key == m_entry_point) {
        // Find new entry point (highest layer node)
        m_entry_point = ObjKey();
        m_entry_point_layer = -1;
        for (const auto& pair : m_vectors) {
            if (pair.second.layer > m_entry_point_layer) {
                m_entry_point = pair.second.obj_key;
                m_entry_point_layer = pair.second.layer;
            }
        }
    }
    
    // Persist changes to storage
    save_to_storage();
}

ObjKey HNSWIndex::find_first(const Mixed& value) const
{
    auto results = search_knn(extract_vector(value), 1);
    return results.empty() ? ObjKey() : results[0].first;
}

void HNSWIndex::find_all(std::vector<ObjKey>& result, Mixed value, bool /*case_insensitive*/) const
{
    // For vector search, "find all" doesn't make sense the same way
    // We'll return top-k results instead
    auto results = search_knn(extract_vector(value), 10);
    result.clear();
    result.reserve(results.size());
    for (const auto& pair : results) {
        result.push_back(pair.first);
    }
}

FindRes HNSWIndex::find_all_no_copy(Mixed value, InternalFindResult& result) const
{
    std::vector<ObjKey> keys;
    find_all(keys, value, false);
    // This would need proper implementation to avoid copying
    return FindRes_not_found;  // Placeholder
}

size_t HNSWIndex::count(const Mixed& /*value*/) const
{
    std::shared_lock lock(m_mutex);
    return m_vectors.size();
}

void HNSWIndex::clear()
{
    std::unique_lock lock(m_mutex);
    m_vectors.clear();
    m_entry_point = ObjKey();
    m_entry_point_layer = -1;
    
    // Persist changes to storage
    save_to_storage();
}

bool HNSWIndex::is_empty() const
{
    std::shared_lock lock(m_mutex);
    return m_vectors.empty();
}

void HNSWIndex::insert_bulk(const ArrayUnsigned* keys, uint64_t key_offset, 
                            size_t num_values, ArrayPayload& values)
{
    std::unique_lock lock(m_mutex);
    // Bulk insertion: insert all values individually
    // Future optimization: batch neighbor search and connection
    for (size_t i = 0; i < num_values; ++i) {
        ObjKey key(keys->get(i) + key_offset);
        Mixed value = values.get_any(i);
        insert(key, value);
    }
}

void HNSWIndex::insert_bulk_list(const ArrayUnsigned* keys, uint64_t key_offset,
                                 size_t num_values, ArrayInteger& ref_array)
{
    std::unique_lock lock(m_mutex);
    // Bulk list insertion: insert all vectors individually
    // Each entry in ref_array points to a list of doubles
    for (size_t i = 0; i < num_values; ++i) {
        ObjKey key(keys->get(i) + key_offset);
        // The actual vector data will be extracted via get_vector_for_key
        insert(key, Mixed());
    }
}

// ===================== K-NN Search =====================

std::vector<std::pair<ObjKey, double>> HNSWIndex::search_knn(const std::vector<double>& query_vector,
                                                               size_t k, size_t ef_search) const
{
    std::shared_lock lock(m_mutex);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (m_vectors.empty() || !m_entry_point) {
        auto t1 = std::chrono::high_resolution_clock::now();
        m_metrics.search_count.fetch_add(1);
        m_metrics.total_search_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return {};
    }
    if (k == 0) {
        return {};
    }
    
    validate_vector_dimension(query_vector);
    
    if (ef_search == 0) {
        ef_search = std::max(m_config.ef_search, k);
    }
    // Clamp ef_search to number of vectors to avoid unnecessary work
    ef_search = std::min<size_t>(ef_search, m_vectors.size());
    k = std::min<size_t>(k, m_vectors.size());
    
    // Start from top layer and traverse down
    ObjKey curr_nearest = m_entry_point;
    for (int lc = m_entry_point_layer; lc > 0; --lc) {
        auto results = search_layer(query_vector, curr_nearest, 1, lc);
        if (!results.empty()) {
            curr_nearest = results[0];
        }
    }
    
    // Search at layer 0 with ef parameter
    auto results = search_layer_with_distances(query_vector, curr_nearest, ef_search, 0);
    
    // Return top k results
    if (results.size() > k) {
        results.resize(k);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    m_metrics.search_count.fetch_add(1);
    m_metrics.total_search_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return results;
}

std::vector<std::pair<ObjKey, double>> HNSWIndex::search_radius(const std::vector<double>& query_vector,
                                                                  double max_distance) const
{
    std::shared_lock lock(m_mutex);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (m_vectors.empty() || !m_entry_point) {
        auto t1 = std::chrono::high_resolution_clock::now();
        m_metrics.radius_search_count.fetch_add(1);
        m_metrics.total_radius_search_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return {};
    }
    if (max_distance < 0) {
        return {};
    }
    
    validate_vector_dimension(query_vector);
    
    // Search with large ef to get many candidates
    size_t ef_large = std::min<size_t>(m_config.ef_search * 2, std::max<size_t>(m_config.ef_search, m_vectors.size()));
    auto results = search_knn(query_vector, m_vectors.size(), ef_large);
    
    // Filter by distance threshold
    std::vector<std::pair<ObjKey, double>> filtered;
    for (const auto& pair : results) {
        if (pair.second <= max_distance) {
            filtered.push_back(pair);
        } else {
            break;  // Results are sorted, so we can stop
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    m_metrics.radius_search_count.fetch_add(1);
    m_metrics.total_radius_search_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return filtered;
}

// ===================== Persistence =====================

void HNSWIndex::load_from_storage()
{
    std::unique_lock lock(m_mutex);
    // Array structure:
    // [0] = metadata array ref
    // [1..N] = node array refs (one per node)
    
    if (!m_array || m_array->size() == 0) {
        return;  // Empty index
    }
    
    // Load metadata
    if (m_array->size() > 0) {
        ref_type metadata_ref = m_array->get_as_ref(0);
        if (metadata_ref) {
            Array metadata(m_array->get_alloc());
            metadata.init_from_ref(metadata_ref);
            if (metadata.size() >= 7) {
                uint64_t version = metadata.get(0);
                REALM_ASSERT(version == k_format_version); // For now enforce match
                m_entry_point = ObjKey(metadata.get(1));
                m_entry_point_layer = static_cast<int>(metadata.get(2));
                m_config.vector_dimension = static_cast<size_t>(metadata.get(3));
                m_config.M = static_cast<size_t>(metadata.get(4));
                m_config.ef_construction = static_cast<size_t>(metadata.get(5));
                m_config.ef_search = static_cast<size_t>(metadata.get(6));
            }
        }
    }
    
    // Load nodes
    for (size_t i = 1; i < m_array->size(); ++i) {
        ref_type node_ref = m_array->get_as_ref(i);
        if (!node_ref) continue;
        
        Array node_array(m_array->get_alloc());
        node_array.init_from_ref(node_ref);
        
        if (node_array.size() < 2) continue;
        
        Node node;
        
        // Load basic node info
        ref_type node_info_ref = node_array.get_as_ref(0);
        if (node_info_ref) {
            Array node_info(m_array->get_alloc());
            node_info.init_from_ref(node_info_ref);
            if (node_info.size() >= 2) {
                node.obj_key = ObjKey(node_info.get(0));
                node.layer = static_cast<int>(node_info.get(1));
            }
        }
        
        // Load vector data
        ref_type vector_ref = node_array.get_as_ref(1);
        if (vector_ref) {
            Array vector_array(m_array->get_alloc());
            vector_array.init_from_ref(vector_ref);
            
            node.vector.reserve(vector_array.size());
            for (size_t j = 0; j < vector_array.size(); ++j) {
                int64_t bits = vector_array.get(j);
                double val;
                std::memcpy(&val, &bits, sizeof(double));
                node.vector.push_back(val);
            }
        }
        
        // Load connections for each layer
        node.connections.resize(node.layer + 1);
        for (int layer = 0; layer <= node.layer && (2 + layer) < static_cast<int>(node_array.size()); ++layer) {
            ref_type conn_ref = node_array.get_as_ref(2 + layer);
            if (conn_ref) {
                Array conn_array(m_array->get_alloc());
                conn_array.init_from_ref(conn_ref);
                
                for (size_t j = 0; j < conn_array.size(); ++j) {
                    node.connections[layer].push_back(ObjKey(conn_array.get(j)));
                }
            }
        }
        
        m_vectors[node.obj_key.value] = std::move(node);
    }
}

void HNSWIndex::save_to_storage()
{
    // Note: Caller must hold m_mutex lock (unique_lock)
    Allocator& alloc = m_array->get_alloc();
    // Build new root array off to the side for atomic-like swap
    auto new_root = std::make_unique<Array>(alloc);
    new_root->create(Array::type_HasRefs);

    // Preserve parent linkage info if present
    ArrayParent* parent = nullptr;
    size_t parent_ndx = 0;
    if (m_array->has_parent()) {
        parent = m_array->get_parent();
        parent_ndx = m_array->get_ndx_in_parent();
    }

    // Metadata
    Array metadata(alloc);
    metadata.create(Array::type_Normal);
    metadata.add(k_format_version);
    metadata.add(m_entry_point.value);
    metadata.add(m_entry_point_layer);
    metadata.add(m_config.vector_dimension);
    metadata.add(m_config.M);
    metadata.add(m_config.ef_construction);
    metadata.add(m_config.ef_search);
    new_root->add(metadata.get_ref());

    // Nodes
    for (const auto& pair : m_vectors) {
        const Node& node = pair.second;
        Array node_array(alloc);
        node_array.create(Array::type_HasRefs);

        Array node_info(alloc);
        node_info.create(Array::type_Normal);
        node_info.add(node.obj_key.value);
        node_info.add(node.layer);
        node_array.add(node_info.get_ref());

        Array vector_array(alloc);
        vector_array.create(Array::type_Normal);
        for (double val : node.vector) {
            int64_t bits;
            std::memcpy(&bits, &val, sizeof(double));
            vector_array.add(bits);
        }
        node_array.add(vector_array.get_ref());

        for (int layer = 0; layer <= node.layer; ++layer) {
            Array conn_array(alloc);
            conn_array.create(Array::type_Normal);
            if (layer < static_cast<int>(node.connections.size())) {
                for (ObjKey neighbor : node.connections[layer]) {
                    conn_array.add(neighbor.value);
                }
            }
            node_array.add(conn_array.get_ref());
        }
        new_root->add(node_array.get_ref());
    }

    // Swap in new array
    m_array->destroy();
    m_array = std::move(new_root);
    m_root_array = m_array.get();
    if (parent) {
        m_array->set_parent(parent, parent_ndx);
        m_array->update_parent();
    }
}

void HNSWIndex::rebuild()
{
    std::unique_lock lock(m_mutex);
    // Collect existing nodes' vectors
    std::vector<Node> nodes;
    nodes.reserve(m_vectors.size());
    for (auto &p : m_vectors) {
        nodes.push_back(p.second);
    }
    m_vectors.clear();
    m_entry_point = ObjKey();
    m_entry_point_layer = -1;
    // Reinsert
    for (const auto &n : nodes) {
        // Reuse stored vector
        insert(n.obj_key, Mixed());
    }
}

// ===================== Verification & Debug =====================

void HNSWIndex::verify() const
{
    // Verify graph integrity
    std::shared_lock lock(m_mutex);
    for (const auto& pair : m_vectors) {
        const Node& node = pair.second;
        
        // Check that all connections are bidirectional
        for (int layer = 0; layer <= node.layer && layer < static_cast<int>(node.connections.size()); ++layer) {
            for (ObjKey neighbor : node.connections[layer]) {
                auto neighbor_it = m_vectors.find(neighbor.value);
                if (neighbor_it != m_vectors.end()) {
                    const Node& neighbor_node = neighbor_it->second;
                    if (layer < static_cast<int>(neighbor_node.connections.size())) {
                        const auto& neighbor_conn = neighbor_node.connections[layer];
                        bool found = std::find(neighbor_conn.begin(), neighbor_conn.end(), node.obj_key) != neighbor_conn.end();
                        REALM_ASSERT_EX(found, node.obj_key.value, neighbor.value, layer);
                    }
                }
            }
        }
        // Degree constraints
        for (int layer = 0; layer <= node.layer && layer < static_cast<int>(node.connections.size()); ++layer) {
            size_t max_conn = (layer == 0) ? m_config.M0 : m_config.M;
            REALM_ASSERT_EX(node.connections[layer].size() <= max_conn + 2, node.obj_key.value, layer); // small slack
        }
    }
}

#ifdef REALM_DEBUG
void HNSWIndex::print() const
{
    std::cout << "HNSW Index Statistics:\n";
    std::cout << "  Vectors: " << m_vectors.size() << "\n";
    std::cout << "  Entry point layer: " << m_entry_point_layer << "\n";
    std::cout << "  Vector dimension: " << m_config.vector_dimension << "\n";
    std::cout << "  M: " << m_config.M << ", M0: " << m_config.M0 << "\n";
    std::cout << "  ef_construction: " << m_config.ef_construction << "\n";
    std::cout << "  ef_search: " << m_config.ef_search << "\n";
    
    // Layer distribution
    std::map<int, size_t> layer_dist;
    for (const auto& pair : m_vectors) {
        layer_dist[pair.second.layer]++;
    }
    std::cout << "  Layer distribution:\n";
    for (const auto& ld : layer_dist) {
        std::cout << "    Layer " << ld.first << ": " << ld.second << " nodes\n";
    }
}
#endif

} // namespace realm
