#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace py = pybind11;

static std::pair<int, float> band_match_u8(
    const uint8_t* L, int LW, int LH,
    const uint8_t* R, int RW, int RH,
    int cx, int cy, int patch, int pad)
{
    int half = std::max(2, patch / 2);
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(LW, cx + half + 1);
    int y1 = std::min(LH, cy + half + 1);
    int W = x1 - x0;
    int H = y1 - y0;
    if (W < 3 || H < 3) return {cx, -1.f};
    const int N = W * H;

    double S_t = 0.0, S2_t = 0.0;
    std::vector<double> T; T.resize(N);
    {
        int k = 0;
        for (int yy = 0; yy < H; ++yy) {
            const uint8_t* p = L + (y0 + yy) * LW + x0;
            for (int xx = 0; xx < W; ++xx, ++k) {
                double v = static_cast<double>(p[xx]);
                T[k] = v;
                S_t  += v;
                S2_t += v * v;
            }
        }
    }
    const double mean_t = S_t / N;
    const double var_t  = std::max(0.0, S2_t - (S_t * S_t) / N);
    if (var_t <= 1e-12) return {cx, -1.f}; 
    const double denom_t = std::sqrt(var_t);

    int rx0 = std::max(0, cx - pad - W / 2);
    int rx1 = std::min(RW - W, cx + pad - W / 2);
    int ry0 = std::max(0, cy - 1);
    int ry1 = std::min(RH - H, cy + 1);
    if (rx1 < rx0 || ry1 < ry0) return {cx, -1.f};

    float best_score = -1.f;
    int   best_cx    = cx;

    for (int y = ry0; y <= ry1; ++y) {
        for (int rx = rx0; rx <= rx1; ++rx) {
            double S_r = 0.0, S2_r = 0.0, sum_rt = 0.0;
            int k = 0;
            for (int yy = 0; yy < H; ++yy) {
                const uint8_t* pr = R + (y + yy) * RW + rx;
                for (int xx = 0; xx < W; ++xx, ++k) {
                    double rv = static_cast<double>(pr[xx]);
                    S_r  += rv;
                    S2_r += rv * rv;
                    sum_rt += rv * T[k];
                }
            }
            const double var_r = std::max(0.0, S2_r - (S_r * S_r) / N);
            if (var_r <= 1e-12) continue;

            const double cross = sum_rt - (S_t * S_r) / N;
            const double score = cross / (std::sqrt(var_r) * denom_t);

            if (static_cast<float>(score) > best_score) {
                best_score = static_cast<float>(score);
                best_cx = rx + W / 2;
            }
        }
    }
    return {best_cx, best_score};
}

py::tuple epipolar_match(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> left,
                         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> right,
                         int cx, int cy, int patch, int pad)
{
    py::buffer_info l = left.request();
    py::buffer_info r = right.request();
    if (l.ndim != 2 || r.ndim != 2) {
        throw std::runtime_error("epipolar_match expects 2D uint8 arrays");
    }
    const uint8_t* L = static_cast<const uint8_t*>(l.ptr);
    const uint8_t* R = static_cast<const uint8_t*>(r.ptr);
    const int LW = static_cast<int>(l.shape[1]);
    const int LH = static_cast<int>(l.shape[0]);
    const int RW = static_cast<int>(r.shape[1]);
    const int RH = static_cast<int>(r.shape[0]);

    auto out = band_match_u8(L, LW, LH, R, RW, RH, cx, cy, patch, pad);
    return py::make_tuple(out.first, out.second);
}

PYBIND11_MODULE(_ncc, m) {
    m.def("epipolar_match", &epipolar_match, "band-limited NCC match (no OpenCV)");
}
