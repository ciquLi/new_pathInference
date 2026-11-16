// PathRadar shortest-path C++ backend.
//
// This intentionally preserves the file contract of gen_false_sample.py:
//   <run>/intermediate/rib/<asn> -> <run>/intermediate/shortest/<asn>.txt
// Downstream SQLite migration, training, and evaluation remain Python.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

std::atomic<unsigned long long> temp_sequence{0};

struct Args {
    int max_limit = 0;
    bool skip_existing = true;
    int workers = 0;
};

[[noreturn]] void usage(const std::string& message = {}) {
    if (!message.empty()) {
        std::cerr << "[error] " << message << '\n';
    }
    std::cerr << "usage: pathradar_shortest_cpp -m <max_limit> "
                 "[--skip-existing 0|1] [--workers <n>]\n";
    std::exit(2);
}

int parse_positive_int(const std::string& value, const std::string& name) {
    try {
        size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed < 1) {
            usage(name + " must be a positive integer");
        }
        return parsed;
    } catch (const std::exception&) {
        usage(name + " must be a positive integer");
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "-m" || flag == "--max_limit") {
            if (++index >= argc) usage(flag + " requires a value");
            args.max_limit = parse_positive_int(argv[index], flag);
        } else if (flag == "--skip-existing") {
            if (++index >= argc) usage(flag + " requires a value");
            const std::string value = argv[index];
            if (value != "0" && value != "1") usage("--skip-existing must be 0 or 1");
            args.skip_existing = value == "1";
        } else if (flag == "--workers") {
            if (++index >= argc) usage(flag + " requires a value");
            args.workers = parse_positive_int(argv[index], flag);
        } else if (flag == "-h" || flag == "--help") {
            usage();
        } else {
            usage("unknown argument: " + flag);
        }
    }
    if (args.max_limit == 0) usage("-m/--max_limit is required");
    if (args.workers == 0) {
        const char* from_env = std::getenv("PATHRADAR_WORKERS");
        args.workers = from_env ? parse_positive_int(from_env, "PATHRADAR_WORKERS") : 10;
    }
    return args;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(delimiter, start);
        const size_t length = (end == std::string::npos ? text.size() : end) - start;
        if (length > 0) result.emplace_back(text.substr(start, length));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

class Topology {
  public:
    int id_for(const std::string& asn) {
        const auto found = ids_.find(asn);
        if (found != ids_.end()) return found->second;
        const int id = static_cast<int>(names_.size());
        ids_.emplace(asn, id);
        names_.push_back(asn);
        provider_.emplace_back();
        peer_.emplace_back();
        customer_.emplace_back();
        return id;
    }

    std::optional<int> find_id(const std::string& asn) const {
        const auto found = ids_.find(asn);
        if (found == ids_.end()) return std::nullopt;
        return found->second;
    }

    const std::string& name(int id) const { return names_.at(id); }
    const std::unordered_set<int>& provider(int id) const { return provider_.at(id); }
    const std::unordered_set<int>& peer(int id) const { return peer_.at(id); }
    const std::unordered_set<int>& customer(int id) const { return customer_.at(id); }

    bool is_customer(int asn1, int asn2) const {
        return customer_.at(asn2).count(asn1) != 0;
    }
    bool is_peer(int asn1, int asn2) const {
        return peer_.at(asn2).count(asn1) != 0;
    }
    bool is_provider(int asn1, int asn2) const {
        return provider_.at(asn2).count(asn1) != 0;
    }

    void load(const fs::path& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open topology: " + path.string());
        std::string line;
        while (std::getline(input, line)) {
            line = trim(line);
            if (line.empty() || line.front() == '#') continue;
            auto fields = split(line, '|');
            if (fields.size() != 3) {
                throw std::runtime_error("invalid topology row: " + line);
            }
            for (std::string& field : fields) field = trim(field);
            const int asn1 = id_for(fields[0]);
            const int asn2 = id_for(fields[1]);
            if (fields[2] == "0" || fields[2] == "1") {
                peer_[asn1].insert(asn2);
                peer_[asn2].insert(asn1);
            } else if (fields[2] == "-1") {
                customer_[asn1].insert(asn2);
                provider_[asn2].insert(asn1);
            }
        }
    }

  private:
    std::unordered_map<std::string, int> ids_;
    std::vector<std::string> names_;
    std::vector<std::unordered_set<int>> provider_;
    std::vector<std::unordered_set<int>> peer_;
    std::vector<std::unordered_set<int>> customer_;
};

struct PathState {
    std::vector<int> nodes;
};

std::string path_key(const std::vector<int>& nodes) {
    std::string key;
    key.reserve(nodes.size() * 5);
    for (const int node : nodes) {
        key.append(std::to_string(node));
        key.push_back(',');
    }
    return key;
}

bool contains_node(const std::vector<int>& path, int node) {
    return std::find(path.begin(), path.end(), node) != path.end();
}

bool numeric_asn_less(const Topology& topology, const PathState& left, const PathState& right) {
    const std::string& left_name = topology.name(left.nodes.back());
    const std::string& right_name = topology.name(right.nodes.back());
    try {
        return std::stoll(left_name) < std::stoll(right_name);
    } catch (const std::exception&) {
        return left_name < right_name;
    }
}

class Generator {
  public:
    Generator(const Topology& topology, fs::path rib_dir, fs::path shortest_dir, int max_limit)
        : topology_(topology), rib_dir_(std::move(rib_dir)), shortest_dir_(std::move(shortest_dir)), max_limit_(max_limit) {}

    void generate(const std::string& origin_name) const {
        const fs::path rib_path = rib_dir_ / origin_name;
        const fs::path output_path = shortest_dir_ / (origin_name + ".txt");
        const fs::path temp_path = output_path.string() + ".tmp." + std::to_string(temp_sequence.fetch_add(1));

        std::unordered_set<std::string> destination_names;
        std::ifstream input;
        for (int attempt = 0; attempt < 3; ++attempt) {
            input.open(rib_path);
            if (input) break;
            input.clear();
            if (attempt == 2) {
                std::cerr << "[warn] missing rib file after retries asn=" << origin_name
                          << " path=" << rib_path << std::endl;
                return;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        {
            std::string line;
            while (std::getline(input, line)) {
                const size_t ampersand = line.find('&');
                const std::string paths = line.substr(0, ampersand);
                std::istringstream path_stream(paths);
                std::string path;
                while (path_stream >> path) {
                    for (const std::string& destination : split(path, '|')) {
                        destination_names.insert(trim(destination));
                    }
                }
            }
        }

        const auto origin_id = topology_.find_id(origin_name);
        std::unordered_map<int, std::vector<PathState>> paths_by_destination;
        if (origin_id.has_value()) {
            paths_by_destination = stream(*origin_id);
        }

        std::ofstream output(temp_path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create output: " + temp_path.string());
        std::vector<std::string> ordered_destinations(destination_names.begin(), destination_names.end());
        for (const std::string& destination_name : ordered_destinations) {
            const auto destination_id = topology_.find_id(destination_name);
            if (!destination_id.has_value() || (origin_id.has_value() && *origin_id == *destination_id)) continue;
            const auto found = paths_by_destination.find(*destination_id);
            if (found == paths_by_destination.end()) continue;
            for (const PathState& path : found->second) {
                for (auto iterator = path.nodes.rbegin(); iterator != path.nodes.rend(); ++iterator) {
                    if (iterator != path.nodes.rbegin()) output << '|';
                    output << topology_.name(*iterator);
                }
                output << '\n';
            }
        }
        output.close();
        if (!output) throw std::runtime_error("failed writing output: " + temp_path.string());
        fs::rename(temp_path, output_path);
    }

  private:
    bool condition(
        const PathState& current,
        int neighbor,
        const std::unordered_map<std::string, int>& path_flags
    ) const {
        if (contains_node(current.nodes, neighbor) || current.nodes.size() >= 9) return false;
        const auto flag = path_flags.find(path_key(current.nodes));
        if (flag == path_flags.end()) {
            throw std::runtime_error("missing valley-free state for active path");
        }
        return flag->second == 0 || topology_.is_customer(neighbor, current.nodes.back());
    }

    bool smaller(int asn1, int asn2, int asn3, int asn4) const {
        if (topology_.is_customer(asn1, asn2) &&
            (topology_.is_peer(asn3, asn4) || topology_.is_provider(asn3, asn4))) {
            return true;
        }
        return topology_.is_peer(asn1, asn2) && topology_.is_provider(asn3, asn4);
    }

    void consider_path(
        std::unordered_map<int, std::vector<PathState>>& paths_by_destination,
        std::unordered_map<int, PathState>& worst,
        std::unordered_map<std::string, PathState>& candidates,
        std::unordered_map<std::string, int>& path_flags,
        const PathState& current,
        int neighbor,
        int& relation_kind) const {
        PathState right_path = current;
        right_path.nodes.push_back(neighbor);
        auto& saved = paths_by_destination[neighbor];
        bool accepted = false;
        int effective_flag = relation_kind;
        if (static_cast<int>(saved.size()) < max_limit_) {
            saved.push_back(right_path);
            path_flags[path_key(right_path.nodes)] = effective_flag;
            accepted = true;
        } else {
            for (size_t index = 0; index < saved.size(); ++index) {
                // Python reuses the outer relation-loop variable here. Preserve its
                // resulting value for both replacement and worst-path state updates.
                effective_flag = static_cast<int>(index);
                if (saved[index].nodes.size() > right_path.nodes.size()) {
                    saved.pop_back();
                    saved.insert(saved.begin() + static_cast<std::ptrdiff_t>(index), right_path);
                    path_flags[path_key(right_path.nodes)] = effective_flag;
                    accepted = true;
                    break;
                }
            }
            relation_kind = effective_flag;
        }
        if (accepted) candidates.emplace(path_key(right_path.nodes), right_path);

        const auto old_worst = worst.find(neighbor);
        if (old_worst == worst.end() || old_worst->second.nodes.empty() ||
            (old_worst->second.nodes.size() >= 2 &&
             smaller(current.nodes.back(), neighbor,
                     old_worst->second.nodes[old_worst->second.nodes.size() - 2],
                     old_worst->second.nodes.back()))) {
            worst[neighbor] = right_path;
            path_flags[path_key(right_path.nodes)] = effective_flag;
            candidates.emplace(path_key(right_path.nodes), right_path);
        }
    }

    std::unordered_map<int, std::vector<PathState>> stream(int origin) const {
        std::unordered_map<int, std::vector<PathState>> paths_by_destination;
        std::unordered_map<int, PathState> worst;
        std::unordered_map<std::string, int> path_flags;
        const PathState origin_path{{origin}};
        std::vector<PathState> active{origin_path};
        paths_by_destination[origin] = {origin_path};
        worst[origin] = origin_path;
        path_flags[path_key(origin_path.nodes)] = 0;

        while (!active.empty()) {
            std::unordered_map<std::string, PathState> candidates;
            for (int relation_index = 0; relation_index < 3; ++relation_index) {
                int state_assignment = relation_index;
                for (const PathState& current : active) {
                    const std::unordered_set<int>* neighbors = nullptr;
                    if (relation_index == 0) neighbors = &topology_.provider(current.nodes.back());
                    if (relation_index == 1) neighbors = &topology_.peer(current.nodes.back());
                    if (relation_index == 2) neighbors = &topology_.customer(current.nodes.back());
                    for (const int neighbor : *neighbors) {
                        if (condition(current, neighbor, path_flags)) {
                            consider_path(
                                paths_by_destination,
                                worst,
                                candidates,
                                path_flags,
                                current,
                                neighbor,
                                state_assignment
                            );
                        }
                    }
                }
            }
            active.clear();
            active.reserve(candidates.size());
            for (auto& [_, candidate] : candidates) active.push_back(std::move(candidate));
            std::sort(active.begin(), active.end(), [this](const PathState& left, const PathState& right) {
                return numeric_asn_less(topology_, left, right);
            });
        }
        return paths_by_destination;
    }

    const Topology& topology_;
    fs::path rib_dir_;
    fs::path shortest_dir_;
    int max_limit_;
};

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const char* run_dir_env = std::getenv("PATHRADAR_RUN_DIR");
        if (!run_dir_env || std::string(run_dir_env).empty()) {
            usage("PATHRADAR_RUN_DIR is required");
        }
        const fs::path intermediate_dir = fs::path(run_dir_env) / "intermediate";
        const fs::path rib_dir = intermediate_dir / "rib";
        const fs::path shortest_dir = intermediate_dir / "shortest";
        if (!fs::is_directory(rib_dir)) {
            throw std::runtime_error("missing rib directory: " + rib_dir.string());
        }
        fs::create_directories(shortest_dir);

        Topology topology;
        topology.load(intermediate_dir / "asrel_rib.txt");

        std::vector<std::string> rib_files;
        for (const fs::directory_entry& entry : fs::directory_iterator(rib_dir)) {
            if (entry.is_regular_file()) rib_files.push_back(entry.path().filename().string());
        }
        std::sort(rib_files.begin(), rib_files.end());

        std::unordered_set<std::string> existing;
        if (args.skip_existing) {
            for (const fs::directory_entry& entry : fs::directory_iterator(shortest_dir)) {
                if (!entry.is_regular_file()) continue;
                const std::string name = entry.path().filename().string();
                if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
                    existing.insert(name.substr(0, name.size() - 4));
                } else {
                    existing.insert(name);
                }
            }
        }
        std::vector<std::string> pending;
        for (const std::string& asn : rib_files) {
            if (!args.skip_existing || existing.count(asn) == 0) pending.push_back(asn);
        }
        std::cout << "[shortest] total_rib=" << rib_files.size()
                  << " pending=" << pending.size()
                  << " skipped=" << (rib_files.size() - pending.size())
                  << " skip_existing=" << (args.skip_existing ? 1 : 0) << std::endl;
        if (pending.empty()) {
            std::cout << "[shortest] nothing to do" << std::endl;
            return 0;
        }

        Generator generator(topology, rib_dir, shortest_dir, args.max_limit);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::string first_error;
        const int thread_count = std::min<int>(args.workers, static_cast<int>(pending.size()));
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (int index = 0; index < thread_count; ++index) {
            threads.emplace_back([&] {
                while (!failed.load(std::memory_order_relaxed)) {
                    const size_t task_index = next.fetch_add(1);
                    if (task_index >= pending.size()) return;
                    try {
                        generator.generate(pending[task_index]);
                    } catch (const std::exception& error) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!failed.exchange(true)) first_error = error.what();
                        return;
                    }
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
        if (failed) throw std::runtime_error(first_error);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[error] " << error.what() << std::endl;
        return 1;
    }
}
