#include "testsettings.hpp"

#include <realm.hpp>
#include <realm/table.hpp>

#include "test.hpp"

using namespace realm;

// Test 1: Simple index creation
TEST(HNSW_Simple_IndexCreation)
{
    Table table;
    auto vec_col = table.add_column_list(type_Double, "vector");
    
    table.add_search_index(vec_col, IndexType::HNSW);
    
    CHECK(table.has_search_index(vec_col));
}

// Test 2: Simple insertion
TEST(HNSW_Simple_Insertion)
{
    Table table;
    auto vec_col = table.add_column_list(type_Double, "vector");
    table.add_search_index(vec_col, IndexType::HNSW);
    
    auto obj = table.create_object();
    auto list = obj.get_list<double>(vec_col);
    list.add(1.0);
    list.add(2.0);
    list.add(3.0);
    
    CHECK_EQUAL(table.size(), 1);
}

// Test 3: Simple k-NN search
TEST(HNSW_Simple_KNN_Search)
{
    Table table;
    auto vec_col = table.add_column_list(type_Double, "vector");
    
    // Insert test vectors BEFORE creating index
    std::vector<std::vector<double>> vectors = {
        {0.0, 0.0, 0.0},
        {0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0}
    };
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        auto obj = table.create_object();
        auto list = obj.get_list<double>(vec_col);
        for (double val : vectors[i]) {
            list.add(val);
        }
    }
    
    // Create index AFTER data is populated
    table.add_search_index(vec_col, IndexType::HNSW);
    
    // Search for k=3 nearest to [0, 0, 0]
    std::vector<double> query = {0.0, 0.0, 0.0};
    Query q = table.where();
    TableView results = q.vector_search_knn(vec_col, query, 3);
    
    // Should return 3 results
    CHECK_EQUAL(results.size(), 3);
}

// Test 4: Radius search
TEST(HNSW_Simple_Radius_Search)
{
    Table table;
    auto vec_col = table.add_column_list(type_Double, "vector");
    
    // Insert vectors at various distances BEFORE creating index
    std::vector<std::vector<double>> vectors = {
        {0.0, 0.0},
        {0.1, 0.0},
        {0.0, 0.2},
        {0.3, 0.4},
        {1.0, 1.0}
    };
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        auto obj = table.create_object();
        auto list = obj.get_list<double>(vec_col);
        for (double val : vectors[i]) {
            list.add(val);
        }
    }
    
    // Create index AFTER data is populated
    table.add_search_index(vec_col, IndexType::HNSW);
    
    // Search within radius 0.3
    std::vector<double> query = {0.0, 0.0};
    Query q = table.where();
    TableView results = q.vector_search_radius(vec_col, query, 0.3);
    
    // Should find at least 3 items within 0.3 (items 0, 1, 2)
    CHECK_GREATER_EQUAL(results.size(), 3);
}
