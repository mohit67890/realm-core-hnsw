/*************************************************************************
 *
 * Simple HNSW Integration Test - Validates Realm Architecture Integration
 *
 **************************************************************************/

#include "test.hpp"
#include "test_table_helper.hpp"

#include <realm/db.hpp>
#include <realm/table.hpp>
#include <realm/query.hpp>
#include <realm/table_view.hpp>
#include <realm/list.hpp>
#include <realm/replication.hpp>
#include <realm/history.hpp>
#include <realm/transaction.hpp>

using namespace realm;
using namespace realm::test_util;

TEST(HNSW_BasicCreation)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    
    // Create HNSW index - this should not throw
    CHECK_NOT(table->has_search_index(vec_col));
    table->add_search_index(vec_col, IndexType::HNSW);
    CHECK(table->has_search_index(vec_col));
    
    tr->commit();
}

TEST(HNSW_BasicInsertAndSearch)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert some 3D vectors
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 2.0);
        list.add(i * 0.5);
    }
    
    tr->commit();
    
    // Search for nearest neighbors
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {5.0, 10.0, 2.5};
    
    // This should return results without crashing
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_GREATER(results.size(), 0);
    CHECK_LESS_EQUAL(results.size(), 5);
}

TEST(HNSW_FilteredSearch)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Documents");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    auto cat_col = table->add_column(type_String, "category");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert vectors with categories
    for (int i = 0; i < 20; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 2.0);
        obj.set(cat_col, (i < 10) ? "A" : "B");
    }
    
    tr->commit();
    
    // Search only in category A
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Documents");
    Query q = table2->where().equal(cat_col, "A");
    std::vector<double> query_vec = {5.0, 10.0};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_GREATER(results.size(), 0);
    
    // Verify all results are from category A
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = table2->get_object(key);
        CHECK_EQUAL(obj.get<String>(cat_col), "A");
    }
}

TEST(HNSW_RadiusSearch)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Points");
    auto vec_col = table->add_column_list(type_Double, "coords");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert vectors
    for (int i = 0; i < 10; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 1.0);
    }
    
    tr->commit();
    
    // Radius search
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Points");
    Query q = table2->where();
    std::vector<double> query_vec = {5.0, 5.0};
    
    auto results = q.vector_search_radius(vec_col, query_vec, 3.0);
    CHECK_GREATER(results.size(), 0);
}

TEST(HNSW_HighDimensional)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Embeddings");
    auto vec_col = table->add_column_list(type_Double, "vector");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert 128-dimensional vectors
    const int dim = 128;
    for (int i = 0; i < 15; i++) {
        auto obj = table->create_object();
        auto list = obj.get_list<double>(vec_col);
        for (int d = 0; d < dim; d++) {
            list.add(std::sin(i + d * 0.1));
        }
    }
    
    tr->commit();
    
    // Search with 128D query
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Embeddings");
    Query q = table2->where();
    
    std::vector<double> query_vec;
    for (int d = 0; d < dim; d++) {
        query_vec.push_back(std::sin(7 + d * 0.1));
    }
    
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_EQUAL(results.size(), 5);
}

TEST(HNSW_Persistence)
{
    SHARED_GROUP_TEST_PATH(path);
    
    // Create and populate
    {
        std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
        auto tr = db->start_write();
        TableRef table = tr->add_table("Vectors");
        auto vec_col = table->add_column_list(type_Double, "data");
        table->add_search_index(vec_col, IndexType::HNSW);
        
        for (int i = 0; i < 10; i++) {
            auto obj = table->create_object();
            auto list = obj.get_list<double>(vec_col);
            list.add(i * 1.0);
            list.add(i * 2.0);
        }
        
        tr->commit();
    }
    
    // Reopen and verify
    {
        std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
        auto tr = db->start_read();
        TableRef table = tr->get_table("Vectors");
        auto vec_col = table->get_column_key("data");
        
        CHECK(table->has_search_index(vec_col));
        
        Query q = table->where();
        std::vector<double> query_vec = {5.0, 10.0};
        auto results = q.vector_search_knn(vec_col, query_vec, 3);
        CHECK_EQUAL(results.size(), 3);
    }
}

TEST(HNSW_MultipleUpdates)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert initial vectors
    std::vector<ObjKey> keys;
    for (int i = 0; i < 5; i++) {
        auto obj = table->create_object();
        keys.push_back(obj.get_key());
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 2.0);
    }
    
    tr->commit();
    
    // Add more vectors in new transaction
    auto tr2 = db->start_write();
    TableRef table2 = tr2->get_table("Vectors");
    
    for (int i = 5; i < 10; i++) {
        auto obj = table2->create_object();
        auto list = obj.get_list<double>(vec_col);
        list.add(i * 1.0);
        list.add(i * 2.0);
    }
    
    tr2->commit();
    
    // Verify search works after multiple updates
    auto tr3 = db->start_read();
    TableRef table3 = tr3->get_table("Vectors");
    Query q = table3->where();
    std::vector<double> query_vec = {5.0, 10.0};
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_EQUAL(results.size(), 5);
}

TEST(HNSW_EmptyTable)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    tr->commit();
    
    // Search on empty table should not crash
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {1.0, 2.0, 3.0};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_EQUAL(results.size(), 0);
}

TEST(HNSW_SingleVector)
{
    SHARED_GROUP_TEST_PATH(path);
    std::unique_ptr<Replication> hist(make_in_realm_history());
    DBRef db = DB::create(*hist, path, DBOptions(crypt_key()));
    
    auto tr = db->start_write();
    TableRef table = tr->add_table("Vectors");
    auto vec_col = table->add_column_list(type_Double, "embedding");
    table->add_search_index(vec_col, IndexType::HNSW);
    
    // Insert single vector
    auto obj = table->create_object();
    auto list = obj.get_list<double>(vec_col);
    list.add(1.0);
    list.add(2.0);
    list.add(3.0);
    
    tr->commit();
    
    // Search should find the one vector
    auto tr2 = db->start_read();
    TableRef table2 = tr2->get_table("Vectors");
    Query q = table2->where();
    std::vector<double> query_vec = {1.1, 2.1, 3.1};
    
    auto results = q.vector_search_knn(vec_col, query_vec, 5);
    CHECK_EQUAL(results.size(), 1);
}
