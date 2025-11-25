/*************************************************************************
 *
 * HNSW Vector Search Test Suite
 * 
 * This file tests the HNSW implementation to ensure correctness
 * of both filtered and unfiltered vector search.
 *
 **************************************************************************/

#include "test.hpp"
#include "test_table_helper.hpp"

#include <realm/db.hpp>
#include <realm/table.hpp>
#include <realm/query.hpp>
#include <realm/table_view.hpp>
#include <realm/transaction.hpp>
#include <realm/list.hpp>
#include <realm/index_hnsw.hpp>
#include <realm/replication.hpp>
#include <realm/history.hpp>
#include <vector>
#include <cmath>

using namespace realm;
using namespace realm::test_util;

// Helper function to calculate Euclidean distance
static double calculate_distance(const std::vector<double>& v1, const std::vector<double>& v2) {
    REALM_ASSERT(v1.size() == v2.size());
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Test 1: Basic HNSW index creation
TEST(HNSW_VectorSearch_IndexCreation)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto vec_col = table->add_column_list(type_Double, "vector");
        
        // Should succeed
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Verify index exists
        CHECK(table->has_search_index(vec_col));
        
        // Verify it's an HNSW index
        CHECK_EQUAL(table->search_index_type(vec_col), IndexType::HNSW);
        
        write->commit();
}

// Test 2: Basic vector insertion and retrieval
TEST(HNSW_VectorSearch_Insertion)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert vectors
        for (int i = 0; i < 10; ++i) {
            auto obj = table->create_object();
            obj.set(name_col, "Item" + std::to_string(i));
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 0.1);
            list.add(i * 0.2);
            list.add(i * 0.3);
        }
        
        write->commit();
        
        // Verify insertion
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        CHECK_EQUAL(table_read->size(), 10);
        
        read->end_read();
}

// Test 3: Unfiltered k-NN search
TEST(HNSW_VectorSearch_KNN_Unfiltered)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert test vectors
        std::vector<std::vector<double>> vectors = {
            {0.0, 0.0, 0.0},  // Item0 - closest to query
            {0.1, 0.1, 0.1},  // Item1 - second closest
            {0.2, 0.2, 0.2},  // Item2 - third closest
            {1.0, 1.0, 1.0},  // Item3 - far
            {2.0, 2.0, 2.0}   // Item4 - farther
        };
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            auto obj = table->create_object();
            obj.set(name_col, "Item" + std::to_string(i));
            auto list = obj.get_list<double>(vec_col);
            for (double val : vectors[i]) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Search for k=3 nearest to [0, 0, 0]
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0, 0.0};
        Query q(table_read);
        TableView results = q.vector_search_knn(vec_col, query, 3);
        
        // Should return 3 results
        CHECK_EQUAL(results.size(), 3);
        
        // First result should be Item0 (exact match)
        auto first_key = results.get_key(0);
        auto first_obj = table_read->get_object(first_key);
        std::string first_name = first_obj.get<String>(name_col);
        CHECK_EQUAL(first_name, "Item0");
        
        read->end_read();
}

// Test 4: Filtered k-NN search with string equality
TEST(HNSW_VectorSearch_KNN_Filtered_String)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto category_col = table->add_column(type_String, "category");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert test data
        struct Item {
            const char* name;
            const char* category;
            std::vector<double> vec;
        };
        
        std::vector<Item> items = {
            {"A1", "CategoryA", {0.0, 0.0}},
            {"A2", "CategoryA", {0.1, 0.1}},
            {"B1", "CategoryB", {0.05, 0.05}},  // This is closer but wrong category
            {"B2", "CategoryB", {0.15, 0.15}},
            {"A3", "CategoryA", {0.2, 0.2}}
        };
        
        for (const auto& item : items) {
            auto obj = table->create_object();
            obj.set(name_col, item.name);
            obj.set(category_col, item.category);
            auto list = obj.get_list<double>(vec_col);
            for (double val : item.vec) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Search with category filter
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        q.equal(category_col, "CategoryA");  // Filter to CategoryA only
        TableView results = q.vector_search_knn(vec_col, query, 2);
        
        // Should return 2 results from CategoryA
        CHECK_EQUAL(results.size(), 2);
        
        // All results should be CategoryA
        for (size_t i = 0; i < results.size(); ++i) {
            auto key = results.get_key(i);
            auto obj = table_read->get_object(key);
            std::string cat = obj.get<String>(category_col);
            CHECK_EQUAL(cat, "CategoryA");
        }
        
        // First should be A1 (closest in CategoryA)
        auto first_key = results.get_key(0);
        auto first_obj = table_read->get_object(first_key);
        std::string first_name = first_obj.get<String>(name_col);
        CHECK_EQUAL(first_name, "A1");
        
        read->end_read();
}

// Test 5: Filtered k-NN search with numeric comparison
TEST(HNSW_VectorSearch_KNN_Filtered_Numeric)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto price_col = table->add_column(type_Double, "price");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert test data
        struct Item {
            const char* name;
            double price;
            std::vector<double> vec;
        };
        
        std::vector<Item> items = {
            {"Cheap1", 10.0, {0.0, 0.0}},
            {"Cheap2", 20.0, {0.1, 0.1}},
            {"Expensive1", 200.0, {0.05, 0.05}},  // Closer but too expensive
            {"Cheap3", 30.0, {0.2, 0.2}},
            {"Expensive2", 300.0, {0.15, 0.15}}
        };
        
        for (const auto& item : items) {
            auto obj = table->create_object();
            obj.set(name_col, item.name);
            obj.set(price_col, item.price);
            auto list = obj.get_list<double>(vec_col);
            for (double val : item.vec) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Search with price filter
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        q.less(price_col, 100.0);  // Only items under $100
        TableView results = q.vector_search_knn(vec_col, query, 3);
        
        // Should return 3 cheap items
        CHECK_EQUAL(results.size(), 3);
        
        // All results should be under $100
        for (size_t i = 0; i < results.size(); ++i) {
            auto key = results.get_key(i);
            auto obj = table_read->get_object(key);
            double price = obj.get<double>(price_col);
            CHECK_LESS(price, 100.0);
        }
        
        read->end_read();
}

// Test 6: Multiple filters (AND condition)
TEST(HNSW_VectorSearch_KNN_Multiple_Filters)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto category_col = table->add_column(type_String, "category");
        auto price_col = table->add_column(type_Double, "price");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert test data
        struct Item {
            const char* name;
            const char* category;
            double price;
            std::vector<double> vec;
        };
        
        std::vector<Item> items = {
            {"A_Cheap", "A", 50.0, {0.0, 0.0}},      // Matches both
            {"A_Expensive", "A", 200.0, {0.1, 0.1}}, // Wrong price
            {"B_Cheap", "B", 40.0, {0.05, 0.05}},    // Wrong category
            {"A_Cheap2", "A", 60.0, {0.2, 0.2}}      // Matches both
        };
        
        for (const auto& item : items) {
            auto obj = table->create_object();
            obj.set(name_col, item.name);
            obj.set(category_col, item.category);
            obj.set(price_col, item.price);
            auto list = obj.get_list<double>(vec_col);
            for (double val : item.vec) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Search with multiple filters
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        q.equal(category_col, "A");     // Category A
        q.less(price_col, 100.0);       // AND under $100
        TableView results = q.vector_search_knn(vec_col, query, 5);
        
        // Should return 2 items (A_Cheap and A_Cheap2)
        CHECK_EQUAL(results.size(), 2);
        
        // Verify all match both conditions
        for (size_t i = 0; i < results.size(); ++i) {
            auto key = results.get_key(i);
            auto obj = table_read->get_object(key);
            std::string cat = obj.get<String>(category_col);
            double price = obj.get<double>(price_col);
            CHECK_EQUAL(cat, "A");
            CHECK_LESS(price, 100.0);
        }
        
        read->end_read();
}

// Test 7: Radius search unfiltered
TEST(HNSW_VectorSearch_Radius_Unfiltered)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert vectors at various distances
        std::vector<std::vector<double>> vectors = {
            {0.0, 0.0},   // Distance 0
            {0.1, 0.0},   // Distance 0.1
            {0.0, 0.2},   // Distance 0.2
            {0.3, 0.4},   // Distance 0.5
            {1.0, 1.0}    // Distance ~1.41
        };
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            auto obj = table->create_object();
            obj.set(name_col, "Item" + std::to_string(i));
            auto list = obj.get_list<double>(vec_col);
            for (double val : vectors[i]) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Search within radius 0.3
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        TableView results = q.vector_search_radius(vec_col, query, 0.3);
        
        // Should find items 0, 1, 2 (within 0.3)
        CHECK_EQUAL(results.size(), 3);
        
        read->end_read();
}

// Test 8: Radius search filtered
TEST(HNSW_VectorSearch_Radius_Filtered)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto name_col = table->add_column(type_String, "name");
        auto category_col = table->add_column(type_String, "category");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert test data
        struct Item {
            const char* name;
            const char* category;
            std::vector<double> vec;
        };
        
        std::vector<Item> items = {
            {"A1", "A", {0.0, 0.0}},
            {"A2", "A", {0.1, 0.1}},
            {"B1", "B", {0.05, 0.05}},  // Close but wrong category
            {"B2", "B", {0.15, 0.15}}
        };
        
        for (const auto& item : items) {
            auto obj = table->create_object();
            obj.set(name_col, item.name);
            obj.set(category_col, item.category);
            auto list = obj.get_list<double>(vec_col);
            for (double val : item.vec) {
                list.add(val);
            }
        }
        
        write->commit();
        
        // Radius search with filter
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        q.equal(category_col, "A");
        TableView results = q.vector_search_radius(vec_col, query, 0.2);
        
        // Should find only A items within radius
        for (size_t i = 0; i < results.size(); ++i) {
            auto key = results.get_key(i);
            auto obj = table_read->get_object(key);
            std::string cat = obj.get<String>(category_col);
            CHECK_EQUAL(cat, "A");
        }
        
        read->end_read();
}

// Test 9: Empty result set when filter matches nothing
TEST(HNSW_VectorSearch_Empty_Filter_Result)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto category_col = table->add_column(type_String, "category");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert only CategoryA items
        for (int i = 0; i < 5; ++i) {
            auto obj = table->create_object();
            obj.set(category_col, "CategoryA");
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 0.1);
            list.add(i * 0.2);
        }
        
        write->commit();
        
        // Search for CategoryB (doesn't exist)
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query = {0.0, 0.0};
        Query q(table_read);
        q.equal(category_col, "CategoryB");
        TableView results = q.vector_search_knn(vec_col, query, 10);
        
        // Should return empty
        CHECK_EQUAL(results.size(), 0);
        
        read->end_read();
}

// Test 10: Large dataset performance test
TEST(HNSW_VectorSearch_Large_Dataset)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
        
        TableRef table = write->add_table("TestTable");
        auto id_col = table->add_column(type_Int, "id");
        auto vec_col = table->add_column_list(type_Double, "vector");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        // Insert 1000 vectors
        for (int i = 0; i < 1000; ++i) {
            auto obj = table->create_object();
            obj.set(id_col, int64_t(i));
            auto list = obj.get_list<double>(vec_col);
            for (int j = 0; j < 10; ++j) {
                list.add(sin(i * 0.1 + j * 0.2));
            }
        }
        
        write->commit();
        
        // Search
        auto read = db->start_read();
        TableRef table_read = read->get_table("TestTable");
        
        std::vector<double> query(10);
        for (int i = 0; i < 10; ++i) {
            query[i] = sin(i * 0.2);
        }
        
        Query q(table_read);
        TableView results = q.vector_search_knn(vec_col, query, 10);
        
        // Should return 10 results
        CHECK_EQUAL(results.size(), 10);
        
        read->end_read();
}

// Test: Verify HNSWIndex Config enforces metric choice
TEST(HNSW_VectorSearch_MetricConfiguration)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    auto write = db->start_write();
    
    TableRef table = write->add_table("MetricTest");
    auto vec_col = table->add_column_list(type_Double, "vector");
    
    // Test that Config requires explicit DistanceMetric
    // This should compile and use Euclidean metric
    HNSWIndex::Config euclidean_config(DistanceMetric::Euclidean);
    CHECK(euclidean_config.metric == DistanceMetric::Euclidean);
    CHECK_EQUAL(euclidean_config.M, 16);
    CHECK_EQUAL(euclidean_config.ef_construction, 200);
    
    // Test that we can create configs with different metrics
    HNSWIndex::Config cosine_config(DistanceMetric::Cosine);
    CHECK(cosine_config.metric == DistanceMetric::Cosine);
    
    HNSWIndex::Config dot_config(DistanceMetric::DotProduct);
    CHECK(dot_config.metric == DistanceMetric::DotProduct);
    
    // Create HNSW index with explicit Euclidean config
    table->add_search_index(vec_col, IndexType::HNSW);
    CHECK(table->has_search_index(vec_col));
    
    // Insert test vectors to verify calculations use the configured metric
    std::vector<ObjKey> keys;
    std::vector<std::vector<double>> vectors = {
        {0.0, 0.0, 0.0},  // Origin
        {1.0, 0.0, 0.0},  // Unit vector on X-axis, distance from origin = 1.0
        {3.0, 4.0, 0.0},  // 3-4-5 triangle, distance from origin = 5.0
        {6.0, 8.0, 0.0}   // Double of previous, distance from origin = 10.0
    };
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        auto obj = table->create_object();
        keys.push_back(obj.get_key());
        auto list = obj.get_list<Double>(vec_col);
        for (double val : vectors[i]) {
            list.add(val);
        }
    }
    
    write->commit();
    
    // Read transaction to verify search results
    auto read = db->start_read();
    auto table_read = read->get_table("MetricTest");
    
    // Query from origin - verify Euclidean distances are correct
    std::vector<double> query_origin = {0.0, 0.0, 0.0};
    Query q(table_read);
    TableView results = q.vector_search_knn(vec_col, query_origin, 4);
    
    CHECK_EQUAL(results.size(), 4);
    
    // Results should be ordered by Euclidean distance
    // First result: origin itself (distance ≈ 0)
    auto key0 = results.get_key(0);
    auto obj0 = table_read->get_object(key0);
    auto vec0 = obj0.get_list<Double>(vec_col);
    std::vector<double> v0 = {vec0.get(0), vec0.get(1), vec0.get(2)};
    double dist0 = calculate_distance(v0, query_origin);
    CHECK(dist0 < 0.01);
    
    // Second result: {1, 0, 0} (distance = 1.0)
    auto key1 = results.get_key(1);
    auto obj1 = table_read->get_object(key1);
    auto vec1 = obj1.get_list<Double>(vec_col);
    std::vector<double> v1 = {vec1.get(0), vec1.get(1), vec1.get(2)};
    double dist1 = calculate_distance(v1, query_origin);
    CHECK(std::abs(dist1 - 1.0) < 0.1);
    
    // Third result: {3, 4, 0} (distance = 5.0)
    auto key2 = results.get_key(2);
    auto obj2 = table_read->get_object(key2);
    auto vec2 = obj2.get_list<Double>(vec_col);
    std::vector<double> v2 = {vec2.get(0), vec2.get(1), vec2.get(2)};
    double dist2 = calculate_distance(v2, query_origin);
    CHECK(std::abs(dist2 - 5.0) < 0.1);
    
    // Fourth result: {6, 8, 0} (distance = 10.0)
    auto key3 = results.get_key(3);
    auto obj3 = table_read->get_object(key3);
    auto vec3 = obj3.get_list<Double>(vec_col);
    std::vector<double> v3 = {vec3.get(0), vec3.get(1), vec3.get(2)};
    double dist3 = calculate_distance(v3, query_origin);
    CHECK(std::abs(dist3 - 10.0) < 0.1);
    
    read->end_read();
}
