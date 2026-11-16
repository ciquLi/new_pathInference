#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <list>
#include <sqlite3.h>

using namespace std;

// ============================================================
// Engine: loads shared data once, computes per-dest BFS
// ============================================================
struct Engine {
    unordered_map<string, vector<string>> graph;
    unordered_map<string, unordered_map<string, int>> rel_dict;
    unordered_map<string, int> freq_dict;
};

// ============================================================
// Valley-free check
// ============================================================
static int get_rel(const Engine &E, const string &as1, const string &as2) {
    auto it1 = E.rel_dict.find(as1);
    if (it1 != E.rel_dict.end()) {
        auto it2 = it1->second.find(as2);
        if (it2 != it1->second.end()) return it2->second;
    }
    auto it2 = E.rel_dict.find(as2);
    if (it2 != E.rel_dict.end()) {
        auto it3 = it2->second.find(as1);
        if (it3 != it2->second.end()) return -it3->second;
    }
    return 2;
}

static bool valley_free(const vector<string> &as_list, const Engine &E) {
    int n = (int)as_list.size();
    vector<int> rels; rels.reserve(n - 1);
    for (int i = 0; i < n - 1; i++)
        rels.push_back(get_rel(E, as_list[i], as_list[i + 1]));

    int p2p_count = 0;
    for (int r : rels) if (r == 0) p2p_count++;
    if (p2p_count > 1) return false;

    for (size_t i = 0; i < rels.size(); i++) {
        if (rels[i] == -1) {
            for (size_t j = i + 1; j < rels.size(); j++)
                if (rels[j] == 1 || rels[j] == 0) return false;
        }
        if (rels[i] == 0) {
            for (size_t j = i + 1; j < rels.size(); j++)
                if (rels[j] == 1) return false;
        }
    }
    return true;
}

// ============================================================
// Path key for sorting
// ============================================================
static int path_first_hop(const vector<string> &p) {
    if (p.size() <= 1) return 0;
    return atoi(p[1].c_str());
}

struct PathSortCtx {
    const unordered_map<string, vector<vector<string>>> *sure_paths;
    const unordered_map<string, int> *freq_dict;
};

static int path_key_ulen(const vector<string> &path, PathSortCtx *ctx) {
    int plen = (int)path.size();
    for (int i = 0; i < plen - 1; i++) {
        auto it = ctx->sure_paths->find(path[i]);
        if (it != ctx->sure_paths->end()) {
            for (const auto &sp : it->second) {
                int slen = (int)sp.size();
                if (plen - i != slen) continue;
                bool match = true;
                for (int k = 0; k < slen; k++)
                    if (sp[k] != path[i + k]) { match = false; break; }
                if (match) return i;
            }
        }
    }
    return 0;
}

static int path_freq(const vector<string> &path, PathSortCtx *ctx) {
    int plen = (int)path.size();
    for (int i = 0; i < plen - 1; i++) {
        auto it = ctx->sure_paths->find(path[i]);
        if (it != ctx->sure_paths->end()) {
            for (const auto &sp : it->second) {
                int slen = (int)sp.size();
                if (plen - i != slen) continue;
                bool match = true;
                for (int k = 0; k < slen; k++)
                    if (sp[k] != path[i + k]) { match = false; break; }
                if (match) {
                    string key;
                    for (int k = i; k < plen; k++) {
                        if (k > i) key += "->";
                        key += path[k];
                    }
                    auto fi = ctx->freq_dict->find(key);
                    return (fi != ctx->freq_dict->end()) ? fi->second : 1;
                }
            }
        }
    }
    return 1;
}

// ============================================================
// JSON helpers (for per-dest data only — small)
// ============================================================
static string json_escape(const string &s) {
    string o; o.reserve(s.size() + 2);
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

static string vec_to_json(const vector<string> &v) {
    string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ",";
        s += "\"" + json_escape(v[i]) + "\"";
    }
    return s + "]";
}

static string vecvec_to_json(const vector<vector<string>> &vv) {
    string s = "[";
    for (size_t i = 0; i < vv.size(); i++) {
        if (i) s += ",";
        s += vec_to_json(vv[i]);
    }
    return s + "]";
}

static void skip_ws(const char *&p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
}

static string parse_string(const char *&p) {
    if (*p != '"') return "";
    p++;
    string s;
    while (*p && *p != '"') {
        if (*p == '\\') { p++; if (*p) s += *p; }
        else s += *p;
        p++;
    }
    if (*p == '"') p++;
    return s;
}

static vector<string> parse_string_array(const char *&p) {
    vector<string> v;
    skip_ws(p);
    if (*p != '[') return v;
    p++; skip_ws(p);
    if (*p == ']') { p++; return v; }
    while (true) {
        v.push_back(parse_string(p));
        skip_ws(p);
        if (*p == ']') { p++; break; }
        if (*p == ',') p++;
    }
    return v;
}

static vector<vector<string>> parse_array2d(const char *&p) {
    vector<vector<string>> v;
    skip_ws(p);
    if (*p != '[') return v;
    p++; skip_ws(p);
    if (*p == ']') { p++; return v; }
    while (true) {
        v.push_back(parse_string_array(p));
        skip_ws(p);
        if (*p == ']') { p++; break; }
        if (*p == ',') p++;
    }
    return v;
}

static unordered_map<string, int> parse_int_dict(const char *&p) {
    unordered_map<string, int> m;
    skip_ws(p);
    if (*p != '{') return m;
    p++; skip_ws(p);
    if (*p == '}') { p++; return m; }
    while (true) {
        string key = parse_string(p);
        skip_ws(p);
        if (*p == ':') p++;
        skip_ws(p);
        int val = 0, neg = 0;
        if (*p == '-') { neg = 1; p++; }
        while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
        if (neg) val = -val;
        m[key] = val;
        skip_ws(p);
        if (*p == '}') { p++; break; }
        if (*p == ',') p++;
    }
    return m;
}

static unordered_map<string, vector<vector<string>>> parse_sure_paths(const char *&p) {
    unordered_map<string, vector<vector<string>>> m;
    skip_ws(p);
    if (*p != '{') return m;
    p++; skip_ws(p);
    if (*p == '}') { p++; return m; }
    while (true) {
        string key = parse_string(p);
        skip_ws(p);
        if (*p == ':') p++;
        m[key] = parse_array2d(p);
        skip_ws(p);
        if (*p == '}') { p++; break; }
        if (*p == ',') p++;
    }
    return m;
}

// ============================================================
// C API
// ============================================================
extern "C" {

void* engine_create(const char *db_path, const char *edge_json_path, const char *rel_json_path) {
    Engine *E = new Engine();

    // 1. Load frequency_dict from SQLite (44M rows, must be fast)
    fprintf(stderr, "[C++] loading freq_dict from %s...\n", db_path);
    {
        sqlite3 *db = nullptr;
        if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK) {
            sqlite3_exec(db, "PRAGMA cache_size=-2000000; PRAGMA mmap_size=26843545600;", nullptr, nullptr, nullptr);
            sqlite3_stmt *stmt = nullptr;
            if (sqlite3_prepare_v2(db, "SELECT path, freq FROM sure_paths", -1, &stmt, nullptr) == SQLITE_OK) {
                E->freq_dict.reserve(46000000);
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    string path((const char*)sqlite3_column_text(stmt, 0));
                    int freq = sqlite3_column_int(stmt, 1);
                    E->freq_dict[move(path)] = freq;
                }
                sqlite3_finalize(stmt);
            }
            sqlite3_close(db);
        }
        fprintf(stderr, "[C++] freq_dict loaded: %zu entries\n", E->freq_dict.size());
    }

    // 2. Load edges from JSON file
    fprintf(stderr, "[C++] loading graph from %s...\n", edge_json_path);
    {
        FILE *fp = fopen(edge_json_path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *buf = new char[sz + 1];
            fread(buf, 1, sz, fp);
            buf[sz] = 0;
            fclose(fp);

            const char *p = buf;
            skip_ws(p);
            if (*p == '{') p++;
            while (*p) {
                skip_ws(p);
                if (*p == '}') { p++; break; }
                string key = parse_string(p);
                skip_ws(p);
                if (*p == ':') p++;
                skip_ws(p);
                // skip int value
                while (*p >= '0' && *p <= '9') p++;
                size_t dash = key.find('-');
                if (dash != string::npos) {
                    string u = key.substr(0, dash), v = key.substr(dash + 1);
                    E->graph[u].push_back(v);
                    E->graph[v].push_back(u);
                }
                skip_ws(p);
                if (*p == ',') p++;
            }
            delete[] buf;
            fprintf(stderr, "[C++] graph loaded: %zu nodes\n", E->graph.size());
        }
    }

    // 3. Load rel_dict from JSON file
    fprintf(stderr, "[C++] loading rel_dict from %s...\n", rel_json_path);
    {
        FILE *fp = fopen(rel_json_path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *buf = new char[sz + 1];
            fread(buf, 1, sz, fp);
            buf[sz] = 0;
            fclose(fp);

            const char *p = buf;
            skip_ws(p);
            if (*p == '{') p++;
            while (*p) {
                skip_ws(p);
                if (*p == '}') { p++; break; }
                string key = parse_string(p);
                skip_ws(p);
                if (*p == ':') p++;
                unordered_map<string, int> inner = parse_int_dict(p);
                E->rel_dict[move(key)] = move(inner);
                skip_ws(p);
                if (*p == ',') p++;
            }
            delete[] buf;
            fprintf(stderr, "[C++] rel_dict loaded: %zu keys\n", E->rel_dict.size());
        }
    }

    return E;
}

void engine_destroy(void *handle) {
    delete (Engine*)handle;
}

// Input:  {"sure_paths": {...}, "base_as": {...}}
// Output: {"src1": [[...],...], "src2": [[...],...], ...}
char* engine_compute(void *handle, const char *dest_json) {
    Engine *E = (Engine*)handle;
    const char *p = dest_json;

    unordered_map<string, vector<vector<string>>> sure_paths;
    unordered_map<string, int> base_as;
    vector<string> base_as_keys;  // preserve JSON key order (matches Python dict insertion order)

    skip_ws(p);
    if (*p == '{') p++;
    while (*p) {
        skip_ws(p);
        if (*p == '}') { p++; break; }
        string field = parse_string(p);
        skip_ws(p);
        if (*p == ':') p++;
        if (field == "sure_paths") sure_paths = parse_sure_paths(p);
        else if (field == "base_as") {
            // Parse preserving key order for deterministic BFS queue
            skip_ws(p);
            if (*p == '{') {
                p++; skip_ws(p);
                if (*p == '}') { p++; }
                else while (true) {
                    string key = parse_string(p);
                    skip_ws(p);
                    if (*p == ':') p++;
                    skip_ws(p);
                    int val = 0, neg = 0;
                    if (*p == '-') { neg = 1; p++; }
                    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
                    if (neg) val = -val;
                    base_as[key] = val;
                    base_as_keys.push_back(key);
                    skip_ws(p);
                    if (*p == '}') { p++; break; }
                    if (*p == ',') p++;
                }
            }
        }
        skip_ws(p);
        if (*p == ',') p++;
    }

    // --- initActiveQueue ---
    unordered_map<string, vector<vector<string>>> rib_in = sure_paths;

    // Sort context — point to rib_in (growing dict), matching Python where
    // env.surePaths IS rib_in, so sort sees newly discovered paths too
    PathSortCtx ctx = {&rib_in, &E->freq_dict};

    // Python dict preserves insertion order — use list+set for FIFO queue
    list<string> queue_list;
    unordered_set<string> queue_set;

    auto sort_key = [&](const vector<string> &a, const vector<string> &b) {
        int la = (int)a.size(), lb = (int)b.size();
        if (la != lb) return la < lb;
        int ua = path_key_ulen(a, &ctx), ub = path_key_ulen(b, &ctx);
        if (ua != ub) return ua < ub;
        int fa = path_freq(a, &ctx), fb = path_freq(b, &ctx);
        if (fa != fb) return fa > fb;
        return path_first_hop(a) < path_first_hop(b);
    };

    for (const string &src : base_as_keys) {
        queue_list.push_back(src);
        queue_set.insert(src);
        auto it = rib_in.find(src);
        if (it != rib_in.end()) {
            stable_sort(it->second.begin(), it->second.end(), sort_key);
        }
    }

    // --- BFS ---
    while (!queue_list.empty()) {
        string u = queue_list.front();
        queue_list.pop_front();
        queue_set.erase(u);

        auto git = E->graph.find(u);
        if (git == E->graph.end()) continue;
        const auto &neighbors = git->second;
        auto rib_u_it = rib_in.find(u);
        if (rib_u_it == rib_in.end() || rib_u_it->second.empty()) continue;
        const vector<string> &p_u = rib_u_it->second[0];

        for (const string &v : neighbors) {
            if (base_as.count(v)) continue;

            vector<string> v_p_u;
            v_p_u.reserve(1 + p_u.size());
            v_p_u.push_back(v);
            v_p_u.insert(v_p_u.end(), p_u.begin(), p_u.end());

            // loop-free check
            {
                unordered_set<string> seen;
                bool loop = false;
                for (const auto &x : v_p_u) {
                    if (seen.count(x)) { loop = true; break; }
                    seen.insert(x);
                }
                if (loop) continue;
            }

            if (!valley_free(v_p_u, *E)) continue;

            auto rit = rib_in.find(v);
            if (rit != rib_in.end()) {
                // Python: path = v_p_u is a no-op (loop variable rebind),
                // but re-sort IS re-run (rib_in has grown, sort order may change)
                // and re-queue on best-change IS checked
                vector<string> old_best = rit->second[0];
                stable_sort(rit->second.begin(), rit->second.end(), sort_key);
                if (old_best != rit->second[0]) {
                    if (!queue_set.count(v)) {
                        queue_list.push_back(v);
                        queue_set.insert(v);
                    }
                }
            } else {
                rib_in[v] = {v_p_u};
                if (!queue_set.count(v)) {
                    queue_list.push_back(v);
                    queue_set.insert(v);
                }
            }
        }
    }

    // Build output
    string out = "{";
    bool first = true;
    for (const auto &kv : rib_in) {
        if (!first) out += ",";
        first = false;
        out += "\"" + json_escape(kv.first) + "\":" + vecvec_to_json(kv.second);
    }
    out += "}";

    char *result = (char*)malloc(out.size() + 1);
    memcpy(result, out.c_str(), out.size() + 1);
    return result;
}

void engine_free_result(char *ptr) {
    free(ptr);
}

} // extern "C"
