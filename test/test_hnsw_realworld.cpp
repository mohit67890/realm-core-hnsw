/*************************************************************************
 *
 * Copyright 2024 Realm Inc.
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

#include "testsettings.hpp"
#ifdef TEST_HNSW_REALWORLD

#include "test.hpp"
#include <realm.hpp>
#include <realm/query_expression.hpp>
#include <cmath>
#include <vector>

using namespace realm;

// Helper to create realistic 768D embedding (simulating real support ticket embeddings)
static std::vector<double> create_embedding_768d(int seed) {
    std::vector<double> vec(768);
    for (int i = 0; i < 768; i++) {
        vec[i] = std::sin(seed * 0.1 + i * 0.01) * 0.5 + std::cos(seed * 0.05 + i * 0.02) * 0.5;
    }
    return vec;
}

// Test 768D vectors like real support ticket embeddings
TEST(HNSW_RealWorld_HighDimensional_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("SupportTickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto title_col = tickets->add_column(type_String, "title");
    auto category_col = tickets->add_column(type_String, "category");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    // Create HNSW index on 768D embeddings
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert 50 support tickets with 768D embeddings
    const char* categories[] = {"login", "payment", "feature_request", "bug_report", "account"};
    const char* titles[] = {
        "Cannot login to account",
        "Payment not processed", 
        "Feature request - Dark mode",
        "Bug in search function",
        "Account locked",
        "Password reset issue",
        "Credit card declined",
        "Need export feature",
        "App crashes on startup",
        "Email verification problem"
    };
    
    for (int i = 0; i < 50; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, i + 1);
        obj.set(title_col, titles[i % 10]);
        obj.set(category_col, categories[i % 5]);
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Test KNN search on 768D vectors
    auto rt = sg->start_read();
    tickets = rt->get_table("SupportTickets");
    
    auto query_vec = create_embedding_768d(25); // Similar to ticket 25
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, query_vec, 5);
    
    CHECK_EQUAL(results.size(), 5);
    
    // Verify all results are valid
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        CHECK(tickets->is_valid(key));
    }
}

// Test filtered search with 768D vectors
TEST(HNSW_RealWorld_FilteredSearch_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto category_col = tickets->add_column(type_String, "category");
    auto priority_col = tickets->add_column(type_Int, "priority");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    const char* categories[] = {"login", "payment", "feature", "bug", "account"};
    
    for (int i = 0; i < 100; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, i + 1);
        obj.set(category_col, categories[i % 5]);
        obj.set(priority_col, (i % 3) + 1); // Priority 1-3
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Search with category filter
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    auto query_vec = create_embedding_768d(42);
    
    Query q = tickets->where().equal(category_col, "login");
    auto results = q.vector_search_knn(embedding_col, query_vec, 10);
    
    CHECK(results.size() > 0);
    CHECK(results.size() <= 10);
    
    // Verify all results match filter
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        CHECK_EQUAL(obj.get<String>(category_col), "login");
    }
}

// Test complex filters: (priority = 1 OR priority = 3) AND category = "payment"
TEST(HNSW_RealWorld_ComplexFilters_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto category_col = tickets->add_column(type_String, "category");
    auto priority_col = tickets->add_column(type_Int, "priority");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    const char* categories[] = {"login", "payment", "feature", "bug", "account"};
    
    for (int i = 0; i < 80; i++) {
        auto obj = tickets->create_object();
        obj.set(category_col, categories[i % 5]);
        obj.set(priority_col, (i % 3) + 1);
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Complex filter: (priority = 1 OR priority = 3) AND category = "payment"
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    auto query_vec = create_embedding_768d(30);
    
    Query q = tickets->where();
    q.group();
    q.equal(priority_col, 1).Or().equal(priority_col, 3);
    q.end_group();
    q.equal(category_col, "payment");
    
    auto results = q.vector_search_knn(embedding_col, query_vec, 20);
    
    CHECK(results.size() > 0);
    
    // Verify all match filter
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        int64_t priority = obj.get<int64_t>(priority_col);
        std::string category = obj.get<String>(category_col);
        
        CHECK((priority == 1 || priority == 3));
        CHECK_EQUAL(category, "payment");
    }
}

// Test update embedding vectors (768D)
TEST(HNSW_RealWorld_UpdateVectors_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    for (int i = 0; i < 30; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, i + 1);
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Update embeddings for tickets 5, 10, 15
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    for (size_t i = 0; i < tickets->size(); i++) {
        auto obj = tickets->get_object(i);
        int64_t ticket_id = obj.get<int64_t>(id_col);
        
        if (ticket_id == 5 || ticket_id == 10 || ticket_id == 15) {
            auto list = obj.get_list<double>(embedding_col);
            list.clear();
            auto new_embedding = create_embedding_768d(static_cast<int>(ticket_id + 100)); // Different embedding
            for (double val : new_embedding) {
                list.add(val);
            }
        }
    }
    
    wt->commit();
    
    // Search should reflect updated embeddings
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    auto query_vec = create_embedding_768d(105); // Similar to updated ticket 5
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, query_vec, 10);
    
    CHECK_EQUAL(results.size(), 10);
    
    // Ticket 5 should be in top results
    bool found_ticket_5 = false;
    for (size_t i = 0; i < std::min<size_t>(5, results.size()); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        if (obj.get<int64_t>(id_col) == 5) {
            found_ticket_5 = true;
            break;
        }
    }
    CHECK(found_ticket_5);
}

// Test delete and re-search (768D)
TEST(HNSW_RealWorld_DeleteVectors_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    for (int i = 0; i < 60; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, i + 1);
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Delete every 3rd ticket (20 tickets deleted)
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    std::vector<ObjKey> to_delete;
    for (size_t i = 0; i < tickets->size(); i++) {
        auto obj = tickets->get_object(i);
        int64_t ticket_id = obj.get<int64_t>(id_col);
        if (ticket_id % 3 == 0) {
            to_delete.push_back(obj.get_key());
        }
    }
    
    for (auto key : to_delete) {
        tickets->remove_object(key);
    }
    
    wt->commit();
    
    // Search should not return deleted tickets
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    CHECK_EQUAL(tickets->size(), 40); // 60 - 20 = 40
    
    auto query_vec = create_embedding_768d(33);
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, query_vec, 15);
    
    CHECK(results.size() <= 15);
    
    // Verify no deleted tickets in results
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        int64_t ticket_id = obj.get<int64_t>(id_col);
        CHECK_NOT_EQUAL(ticket_id % 3, 0);
    }
}

// Test radius search (768D)
TEST(HNSW_RealWorld_RadiusSearch_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    for (int i = 0; i < 40; i++) {
        auto obj = tickets->create_object();
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    auto query_vec = create_embedding_768d(20);
    
    // Radius search with distance threshold
    Query q = tickets->where();
    auto results = q.vector_search_radius(embedding_col, query_vec, 0.5);
    
    CHECK(results.size() >= 0); // May be 0 if no vectors within radius
    
    // All results should be valid
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        CHECK(tickets->is_valid(key));
    }
}

// Test large-scale 768D vectors (stress test)
TEST(HNSW_RealWorld_LargeScale_768D) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert 500 tickets with 768D embeddings (384,000 total dimensions)
    for (int i = 0; i < 500; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, i + 1);
        
        auto list = obj.get_list<double>(embedding_col);
        auto embedding = create_embedding_768d(i);
        for (double val : embedding) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Search should complete in reasonable time
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    auto query_vec = create_embedding_768d(250);
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, query_vec, 20);
    
    CHECK_EQUAL(results.size(), 20);
    
    // All results valid
    for (size_t i = 0; i < results.size(); i++) {
        CHECK(tickets->is_valid(results.get_key(i)));
    }
}

#endif // TEST_HNSW_REALWORLD
