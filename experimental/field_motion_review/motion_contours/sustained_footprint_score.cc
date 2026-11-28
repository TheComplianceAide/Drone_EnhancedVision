#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {
using Histogram = std::array<int, 256>;

struct Ranks {
    int low;
    int high;
    double fraction;
};

Ranks ranks(int count, double quantile) {
    const double index = (count - 1) * quantile;
    const int low = static_cast<int>(index);
    return {low, std::min(low + 1, count - 1), index - low};
}

double interpolate(int a, int b, double fraction) {
    const int difference = b - a;
    return fraction >= .5 ? b - difference * (1. - fraction) : a + difference * fraction;
}

template <std::size_t N>
std::array<double, N> quantiles(const Histogram& histogram, int count,
                               const std::array<double, N>& quantile) {
    std::array<double, N> values{};
    int value = 0;
    int total = histogram[0];
    for (std::size_t i = 0; i < N; ++i) {
        const Ranks rank = ranks(count, quantile[i]);
        while (total <= rank.low) {
            total += histogram[++value];
        }
        const int low = value;
        while (total <= rank.high) {
            total += histogram[++value];
        }
        values[i] = interpolate(low, value, rank.fraction);
    }
    return values;
}
}

// Obtain exactly the original component statistics from the same luma, response and labels.
// Pixel arithmetic and output ordering match the original component-by-component calculation.
extern "C" int score_footprints(const uint8_t* gray, const uint8_t* core, const uint8_t* response,
                                 const int32_t* labels, const int32_t* stats, int width, int height,
                                 int label_count, double* output) {
    int accepted = 0;
    for (int id = 1; id < label_count; ++id) {
        const int32_t* component = stats + 5 * id;
        const int x = component[0], y = component[1];
        const int w = component[2], h = component[3], area = component[4];
        const auto [small, large] = std::minmax(w, h);
        if (area < 4 || area > 1200 || small < 2 || large > 65 || large > 5 * small ||
            std::min({x, y, width - x - w, height - y - h}) < 9 || area < .20 * w * h) {
            continue;
        }
        Histogram subject{};
        int supported = 0;
        int mass = 0;
        int weighted_x = 0;
        int weighted_y = 0;
        for (int dy = 0; dy < h; ++dy) {
            const int row = (y + dy) * width + x;
            for (int dx = 0; dx < w; ++dx) {
                const int pixel = row + dx;
                if (labels[pixel] == id && core[pixel]) {
                    const int weight = response[pixel];
                    ++subject[gray[pixel]];
                    ++supported;
                    mass += weight;
                    weighted_x += dx * weight;
                    weighted_y += dy * weight;
                }
            }
        }
        if (supported < 4) {
            continue;
        }
        Histogram rim{};
        for (int dy = -7; dy < h + 7; ++dy) {
            const int row = (y + dy) * width + x;
            for (int dx = -7; dx < w + 7; ++dx) {
                if (dy < 0 || dy >= h || dx < 0 || dx >= w) {
                    ++rim[gray[row + dx]];
                }
            }
        }
        const int rim_count = 14 * (w + h + 14);
        const auto [low, middle, high] = quantiles(rim, rim_count, std::array{.15, .50, .85});
        const auto [median] = quantiles(subject, supported, std::array{.50});
        const double strength = middle - median;
        if (strength < std::max(11., 1.12 * (high - low)) || high < 35 || low > 235) {
            continue;
        }
        double* record = output + 8 * accepted++;
        record[0] = x + static_cast<double>(weighted_x) / mass;
        record[1] = y + static_cast<double>(weighted_y) / mass;
        record[2] = x;
        record[3] = y;
        record[4] = w;
        record[5] = h;
        record[6] = strength;
        record[7] = area;
    }
    return accepted;
}
