#include "testsettings.hpp"
#ifdef TEST_HNSW_REALWORLD

#include "test.hpp"
#include <realm.hpp>
#include <realm/query_expression.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace realm;

// Simple JSON parser for our specific format
static std::vector<std::vector<double>> load_embeddings_from_json(const std::string& filename, int max_tickets = 10) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return {}; // Return empty if file not found
    }
    
    std::vector<std::vector<double>> embeddings;
    std::string line;
    bool in_embedding = false;
    std::vector<double> current_embedding;
    
    while (std::getline(file, line)) {
        // Find start of embedding array
        if (line.find("\"embedding\": [") != std::string::npos) {
            in_embedding = true;
            current_embedding.clear();
            continue;
        }
        
        // Parse numbers in embedding
        if (in_embedding) {
            // Check for end of array
            if (line.find("]") != std::string::npos && line.find(",") == std::string::npos) {
                if (!current_embedding.empty()) {
                    embeddings.push_back(current_embedding);
                    if (embeddings.size() >= max_tickets) {
                        break;
                    }
                }
                in_embedding = false;
                continue;
            }
            
            // Extract number from line like "        -0.10983941704034805,"
            size_t start = line.find_first_not_of(" \t");
            if (start != std::string::npos && (line[start] == '-' || std::isdigit(line[start]))) {
                std::string num_str = line.substr(start);
                // Remove trailing comma
                if (num_str.back() == ',') {
                    num_str.pop_back();
                }
                try {
                    double val = std::stod(num_str);
                    current_embedding.push_back(val);
                } catch (...) {
                    // Skip invalid numbers
                }
            }
        }
    }
    
    file.close();
    return embeddings;
}

// Test loading actual JSON embeddings
TEST(HNSW_RealWorld_LoadActualJSON) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    // Load real embeddings from JSON file
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        // File not found - skip test gracefully
        return;
    }
    
    CHECK(embeddings.size() == 10);
    CHECK(embeddings[0].size() == 768); // Should be 768D
    
    // Create database with real embeddings
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert real embeddings
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Test KNN search with real embedding
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, embeddings[0], 5);
    
    CHECK(results.size() > 0);
    CHECK(results.size() <= 5);
    
    // First result should be ticket 1 (identical embedding)
    auto first_key = results.get_key(0);
    auto first_obj = tickets->get_object(first_key);
    CHECK_EQUAL(first_obj.get<int64_t>(id_col), 1);
}

// Test with all 10 tickets + 5 queries from JSON
TEST(HNSW_RealWorld_FullJSONTest) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 15); // 10 tickets + 5 queries
    
    if (embeddings.size() < 10) {
        return; // Skip if file not available
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert first 10 embeddings (tickets)
    for (size_t i = 0; i < 10; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Search with query embeddings (if available)
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    for (size_t query_idx = 10; query_idx < std::min(embeddings.size(), size_t(15)); query_idx++) {
        Query q = tickets->where();
        auto results = q.vector_search_knn(embedding_col, embeddings[query_idx], 3);
        
        CHECK(results.size() > 0);
        CHECK(results.size() <= 3);
        
        // All results should be valid tickets
        for (size_t i = 0; i < results.size(); i++) {
            auto key = results.get_key(i);
            CHECK(tickets->is_valid(key));
        }
    }
}

// Test filtered search with real JSON data
TEST(HNSW_RealWorld_JSON_WithFilters) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        return;
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto category_col = tickets->add_column(type_String, "category");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    const char* categories[] = {"login", "payment", "feature", "bug", "account"};
    
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        obj.set(category_col, categories[i % 5]);
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Filtered search: only "login" category
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    Query q = tickets->where().equal(category_col, "login");
    auto results = q.vector_search_knn(embedding_col, embeddings[0], 5);
    
    CHECK(results.size() > 0);
    
    // Verify all results match filter
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        CHECK_EQUAL(obj.get<String>(category_col), "login");
    }
}

// Test DELETE operations with real JSON data
TEST(HNSW_RealWorld_JSON_Delete) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        return;
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert all 10 real embeddings
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // DELETE tickets 2, 5, 8 (3 deletions)
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    std::vector<ObjKey> to_delete;
    for (size_t i = 0; i < tickets->size(); i++) {
        auto obj = tickets->get_object(i);
        int64_t ticket_id = obj.get<int64_t>(id_col);
        if (ticket_id == 2 || ticket_id == 5 || ticket_id == 8) {
            to_delete.push_back(obj.get_key());
        }
    }
    
    for (auto key : to_delete) {
        tickets->remove_object(key);
    }
    
    wt->commit();
    
    // Verify deleted tickets are gone
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    CHECK_EQUAL(tickets->size(), 7); // 10 - 3 = 7
    
    // Search should not return deleted tickets
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, embeddings[0], 10);
    
    CHECK(results.size() <= 7);
    
    // Verify tickets 2, 5, 8 are NOT in results
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        int64_t ticket_id = obj.get<int64_t>(id_col);
        CHECK(ticket_id != 2 && ticket_id != 5 && ticket_id != 8);
    }
}

// Test DELETE + RE-INSERT with real JSON data
TEST(HNSW_RealWorld_JSON_DeleteAndReinsert) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        return;
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // DELETE ticket 3
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    ObjKey key_to_delete;
    for (size_t i = 0; i < tickets->size(); i++) {
        auto obj = tickets->get_object(i);
        if (obj.get<int64_t>(id_col) == 3) {
            key_to_delete = obj.get_key();
            break;
        }
    }
    
    tickets->remove_object(key_to_delete);
    CHECK_EQUAL(tickets->size(), 9);
    
    wt->commit();
    
    // RE-INSERT ticket 3 with same ID but at end
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    auto new_obj = tickets->create_object();
    new_obj.set(id_col, static_cast<int64_t>(3));
    
    auto list = new_obj.get_list<double>(embedding_col);
    for (double val : embeddings[2]) { // Re-use ticket 3's original embedding
        list.add(val);
    }
    
    wt->commit();
    
    // Search should find re-inserted ticket 3
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    CHECK_EQUAL(tickets->size(), 10);
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, embeddings[2], 5);
    
    CHECK(results.size() > 0);
    
    // Ticket 3 should be in results
    bool found_ticket_3 = false;
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        if (obj.get<int64_t>(id_col) == 3) {
            found_ticket_3 = true;
            break;
        }
    }
    CHECK(found_ticket_3);
}

// Test REMOVE ALL then re-populate with real JSON data
TEST(HNSW_RealWorld_JSON_RemoveAll) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        return;
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    // Insert all 10
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // REMOVE ALL tickets
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    tickets->clear();
    CHECK_EQUAL(tickets->size(), 0);
    
    wt->commit();
    
    // Search on empty table should return 0 results
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    Query q = tickets->where();
    auto results = q.vector_search_knn(embedding_col, embeddings[0], 10);
    
    CHECK_EQUAL(results.size(), 0);
    
    // RE-POPULATE with first 5 tickets
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    for (size_t i = 0; i < 5; i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // Search should work again
    rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    CHECK_EQUAL(tickets->size(), 5);
    
    q = tickets->where();
    results = q.vector_search_knn(embedding_col, embeddings[0], 3);
    
    CHECK(results.size() > 0);
    CHECK(results.size() <= 3);
}

// Test DELETE with filters using real JSON data
TEST(HNSW_RealWorld_JSON_DeleteWithFilter) {
    SHARED_GROUP_TEST_PATH(path);
    DBRef sg = DB::create(make_in_realm_history(), path);
    
    auto embeddings = load_embeddings_from_json("../z_embeddings_data.json", 10);
    
    if (embeddings.empty()) {
        return;
    }
    
    auto wt = sg->start_write();
    auto tickets = wt->add_table("Tickets");
    auto id_col = tickets->add_column(type_Int, "ticket_id");
    auto status_col = tickets->add_column(type_String, "status");
    auto embedding_col = tickets->add_column_list(type_Double, "embedding");
    
    tickets->add_search_index(embedding_col, IndexType::HNSW);
    
    const char* statuses[] = {"active", "pending", "closed"};
    
    for (size_t i = 0; i < embeddings.size(); i++) {
        auto obj = tickets->create_object();
        obj.set(id_col, static_cast<int64_t>(i + 1));
        obj.set(status_col, statuses[i % 3]);
        
        auto list = obj.get_list<double>(embedding_col);
        for (double val : embeddings[i]) {
            list.add(val);
        }
    }
    
    wt->commit();
    
    // DELETE all "closed" tickets
    wt = sg->start_write();
    tickets = wt->get_table("Tickets");
    
    std::vector<ObjKey> to_delete;
    for (size_t i = 0; i < tickets->size(); i++) {
        auto obj = tickets->get_object(i);
        if (obj.get<String>(status_col) == "closed") {
            to_delete.push_back(obj.get_key());
        }
    }
    
    for (auto key : to_delete) {
        tickets->remove_object(key);
    }
    
    wt->commit();
    
    // Search with filter: only "active"
    auto rt = sg->start_read();
    tickets = rt->get_table("Tickets");
    
    Query q = tickets->where().equal(status_col, "active");
    auto results = q.vector_search_knn(embedding_col, embeddings[0], 10);
    
    CHECK(results.size() > 0);
    
    // All results should be "active", none "closed"
    for (size_t i = 0; i < results.size(); i++) {
        auto key = results.get_key(i);
        auto obj = tickets->get_object(key);
        std::string status = obj.get<String>(status_col);
        CHECK_EQUAL(status, "active");
        CHECK_NOT_EQUAL(status, "closed");
    }
}

#endif // TEST_HNSW_REALWORLD
