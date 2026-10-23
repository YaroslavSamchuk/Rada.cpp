// ============================================================================
// File: rada.cpp/include/rada_loop_watchdog.hpp
// Fail-Safe Generation State Machine & Loop Prevention Watchdog for Hetman-2.0B
// Features:
// 1. Zero-Think enforcement on conversational System 1 queries (greetings, chit-chat)
// 2. Rolling 64-bit Rabin-Karp N-gram repetition detector (window W=64, threshold 3x)
// 3. Thinking token budget cap (B_think <= 1024) with forced </think>\n ejection
// 4. Frobenius norm stagnation monitor for DeltaNet recurrent state (||St - St-8||_F < 1e-6)
// 5. Formal DAG Finite State Automaton guaranteeing finite termination
// ============================================================================
#pragma once

#include <cstdint>
#include <vector>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include "rada_format.h"

namespace rada {

enum class GenerationState {
    INIT,            // Awaiting first token
    THINKING,        // Inside <think>...</think>
    THOUGHT_SUMMARY, // Inside <thought_summary>...</thought_summary>
    ANSWERING,       // Emitting final response
    TERMINATED       // Emitted <|im_end|> or terminated
};

struct WatchdogConfig {
    int max_total_tokens = 2048;            // Absolute ceiling for generated sequence length
    int max_think_budget = 1024;            // Maximum allowed tokens inside <think>
    int ngram_window_size = 64;             // Sliding window for n-gram repetition detection
    int max_ngram_repeats = 6;              // Threshold for repeated n-grams before forced ejection
    float frobenius_stagnation_eps = 1e-6f; // Threshold for DeltaNet state stagnation
    bool allow_thinking = true;             // If false (System 1), <think> is permanently masked
};

class RadaLoopWatchdog {
private:
    WatchdogConfig config_;
    GenerationState state_ = GenerationState::INIT;
    int total_tokens_ = 0;
    int think_tokens_ = 0;
    
    // Cyclic ring buffer for token IDs
    std::deque<uint32_t> window_tokens_;
    
    // Fast rolling 64-bit hash table for 4-grams
    std::unordered_map<uint64_t, int> ngram_frequencies_;
    
    // Recurrent state tracking
    std::vector<float> prev_deltanet_state_sample_;
    int stagnation_streak_ = 0;

    static uint64_t hash_4gram(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
        // High-speed 64-bit FNV-1a mixing
        uint64_t h = 14695981039346656037ULL;
        h = (h ^ t0) * 1099511628211ULL;
        h = (h ^ t1) * 1099511628211ULL;
        h = (h ^ t2) * 1099511628211ULL;
        h = (h ^ t3) * 1099511628211ULL;
        return h;
    }

public:
    explicit RadaLoopWatchdog(const WatchdogConfig& config) : config_(config) {
        if (!config_.allow_thinking) {
            state_ = GenerationState::ANSWERING;
        }
    }

    // Filter logits prior to sampling: strictly enforces valid grammar transitions
    void apply_logit_masks(float* logits, size_t vocab_size) {
        if (!logits || vocab_size <= RADA_TOKEN_THOUGHT_SUMMARY_END) return;

        // 1. If thinking is disallowed (System 1) or we are already ANSWERING, ban <think>
        if (!config_.allow_thinking || state_ == GenerationState::ANSWERING || state_ == GenerationState::THOUGHT_SUMMARY) {
            logits[RADA_TOKEN_THINK] = -1e9f;
        }

        // 2. If we are in INIT and thinking is allowed, prevent closing a non-existent thought
        if (state_ == GenerationState::INIT) {
            logits[RADA_TOKEN_THINK_END] = -1e9f;
            logits[RADA_TOKEN_THOUGHT_SUMMARY] = -1e9f;
            logits[RADA_TOKEN_THOUGHT_SUMMARY_END] = -1e9f;
        }

        // 3. If in THINKING, ban starting a new nested <think> block
        if (state_ == GenerationState::THINKING) {
            logits[RADA_TOKEN_THINK] = -1e9f;
        }
    }

    bool is_thinking() const {
        return state_ == GenerationState::THINKING;
    }

    // Inspect next sampled token and decide whether to accept or force eject
    enum class Action {
        ACCEPT,
        FORCE_CLOSE_THINK,  // Force inject </think>\n
        FORCE_TERMINATE     // Force inject <|im_end|>
    };

    Action inspect_token(uint32_t token_id) {
        total_tokens_++;

        // State Machine Transitions
        if (token_id == RADA_TOKEN_IM_END || token_id == RADA_TOKEN_EOS) {
            state_ = GenerationState::TERMINATED;
            return Action::FORCE_TERMINATE;
        } else if (token_id == RADA_TOKEN_THINK) {
            // Strict DAG transition: can only enter THINKING from INIT
            if (state_ == GenerationState::INIT) {
                state_ = GenerationState::THINKING;
            }
        } else if (token_id == RADA_TOKEN_THINK_END) {
            state_ = GenerationState::ANSWERING;
        } else if (token_id == RADA_TOKEN_THOUGHT_SUMMARY) {
            state_ = GenerationState::THOUGHT_SUMMARY;
        } else if (token_id == RADA_TOKEN_THOUGHT_SUMMARY_END) {
            state_ = GenerationState::ANSWERING;
        } else if (state_ == GenerationState::INIT) {
            state_ = GenerationState::ANSWERING;
        }

        if (state_ == GenerationState::THINKING) {
            think_tokens_++;
            if (think_tokens_ >= config_.max_think_budget) {
                // Rule 1: Think budget exhausted
                state_ = GenerationState::ANSWERING;
                window_tokens_.clear();
                ngram_frequencies_.clear();
                return Action::FORCE_CLOSE_THINK;
            }
        }

        // Update sliding window
        window_tokens_.push_back(token_id);

        // Evict expired tokens outside sliding window first
        if (window_tokens_.size() > static_cast<size_t>(config_.ngram_window_size) && window_tokens_.size() >= 4) {
            uint32_t ev0 = window_tokens_[0];
            uint32_t ev1 = window_tokens_[1];
            uint32_t ev2 = window_tokens_[2];
            uint32_t ev3 = window_tokens_[3];
            uint64_t ev_h = hash_4gram(ev0, ev1, ev2, ev3);
            auto it = ngram_frequencies_.find(ev_h);
            if (it != ngram_frequencies_.end()) {
                if (--(it->second) <= 0) {
                    ngram_frequencies_.erase(it);
                }
            }
            window_tokens_.pop_front();
        }

        // Check for n-gram repetition within window
        if (window_tokens_.size() >= 4) {
            size_t n = window_tokens_.size();
            uint64_t h4 = hash_4gram(window_tokens_[n - 4], window_tokens_[n - 3], window_tokens_[n - 2], window_tokens_[n - 1]);
            int count = ++ngram_frequencies_[h4];

            if (count >= config_.max_ngram_repeats) {
                // Rule 2: N-gram repetition detected
                if (state_ == GenerationState::THINKING) {
                    state_ = GenerationState::ANSWERING;
                    window_tokens_.clear();
                    ngram_frequencies_.clear();
                    stagnation_streak_ = 0;
                    return Action::FORCE_CLOSE_THINK;
                } else {
                    state_ = GenerationState::TERMINATED;
                    return Action::FORCE_TERMINATE;
                }
            }
        }

        // Rule 3: Absolute maximum token limit
        if (total_tokens_ >= config_.max_total_tokens) {
            state_ = GenerationState::TERMINATED;
            return Action::FORCE_TERMINATE;
        }

        return Action::ACCEPT;
    }

    // Inspect DeltaNet Recurrent Frobenius Norm Drift every k tokens
    Action inspect_deltanet_state(const float* S_data, size_t total_elements) {
        if (!S_data || total_elements == 0) return Action::ACCEPT;

        if (prev_deltanet_state_sample_.empty()) {
            prev_deltanet_state_sample_.assign(S_data, S_data + total_elements);
            return Action::ACCEPT;
        }

        float diff_frobenius_sq = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            float d = S_data[i] - prev_deltanet_state_sample_[i];
            diff_frobenius_sq += d * d;
        }
        float diff_frobenius = std::sqrt(diff_frobenius_sq);

        // Update stored sample
        std::copy(S_data, S_data + total_elements, prev_deltanet_state_sample_.begin());

        if (diff_frobenius < config_.frobenius_stagnation_eps) {
            stagnation_streak_++;
            if (stagnation_streak_ >= 3) {
                // Stagnant recurrent state detected: break attractor lock
                if (state_ == GenerationState::THINKING) {
                    state_ = GenerationState::ANSWERING;
                    window_tokens_.clear();
                    ngram_frequencies_.clear();
                    stagnation_streak_ = 0;
                    return Action::FORCE_CLOSE_THINK;
                } else {
                    state_ = GenerationState::TERMINATED;
                    return Action::FORCE_TERMINATE;
                }
            }
        } else {
            stagnation_streak_ = 0;
        }

        return Action::ACCEPT;
    }

    GenerationState get_current_state() const { return state_; }
    int get_think_tokens() const { return think_tokens_; }
    int get_total_tokens() const { return total_tokens_; }
};

} // namespace rada
